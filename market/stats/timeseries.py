import warnings

import numpy as np
import pandas as pd
from scipy import stats as st
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import acf, adfuller, coint, grangercausalitytests, kpss, pacf

from market.stats.core import pair, result, verdict
from market.stats.regression import fit


def adf(s: pd.Series) -> pd.Series:
    stat, p, *_ = adfuller(s.dropna(), autolag="AIC", result_object=False)
    return result("ADF", stat, p, "unit root (non-stationary)")


def kpss_test(s: pd.Series) -> pd.Series:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        stat, p, *_ = kpss(s.dropna(), regression="c", nlags="auto")
    return result("KPSS", stat, p, "stationary")


def stationarity(s: pd.Series) -> pd.DataFrame:
    return pd.DataFrame([adf(s), kpss_test(s)])


def autocorr(s: pd.Series, nlags: int = 20) -> pd.DataFrame:
    s = s.dropna()
    return pd.DataFrame({"lag": range(nlags + 1), "acf": acf(s, nlags=nlags), "pacf": pacf(s, nlags=nlags),
                         "bound": 1.96 / np.sqrt(len(s))})


def ljung_box(s: pd.Series, lags: int = 10) -> pd.Series:
    lb = acorr_ljungbox(s.dropna(), lags=[lags])
    return result(f"Ljung–Box ({lags} lags)", lb.lb_stat.iloc[0], lb.lb_pvalue.iloc[0], "no autocorrelation")


def hurst(level: pd.Series, max_lag: int = 50) -> float:
    """Hurst exponent of a level series (e.g. log price): <0.5 mean-reverting, 0.5 random walk, >0.5 trending."""
    x = level.dropna().to_numpy()
    lags = np.arange(2, min(max_lag, len(x) // 2))
    tau = [np.sqrt(np.mean((x[lag:] - x[:-lag]) ** 2)) for lag in lags]
    return float(np.polyfit(np.log(lags), np.log(tau), 1)[0])


def variance_ratio(r: pd.Series, q: int = 5) -> pd.Series:
    """Lo–MacKinlay variance ratio (homoskedastic): VR≈1 for a random walk."""
    r = r.dropna().to_numpy()
    n = len(r)
    vr = pd.Series(r).rolling(q).sum().dropna().var() / (q * r.var(ddof=1))
    z = (vr - 1) / np.sqrt(2 * (2 * q - 1) * (q - 1) / (3 * q * n))
    return result(f"Variance ratio (q={q})", vr, 2 * (1 - st.norm.cdf(abs(z))), "random walk (VR = 1)", z=z)


def cointegration(a: pd.Series, b: pd.Series) -> pd.Series:
    """Engle–Granger test on two price series, with hedge ratio and spread half-life."""
    ab = pair(a, b)
    stat, p, _ = coint(ab.a, ab.b)
    hedge = fit(ab.a, ab.b).params.iloc[1]
    spread = ab.a - hedge * ab.b
    lagged = spread.shift(1)
    k = fit(spread.diff(), lagged).params.iloc[1]
    half_life = -np.log(2) / k if k < 0 else np.inf
    return result("Engle–Granger cointegration", stat, p, "no cointegration", hedge_ratio=hedge, half_life=half_life)


def spread(a: pd.Series, b: pd.Series, window: int = 60) -> pd.DataFrame:
    ab = pair(a, b)
    hedge = fit(ab.a, ab.b).params.iloc[1]
    s = ab.a - hedge * ab.b
    return pd.DataFrame({"spread": s, "zscore": (s - s.rolling(window).mean()) / s.rolling(window).std()})


def granger(effect: pd.Series, cause: pd.Series, maxlag: int = 5) -> pd.DataFrame:
    """Does `cause` help predict `effect`? One row per lag."""
    data = pair(effect, cause)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tests = grangercausalitytests(data[["a", "b"]], maxlag=maxlag)
    rows = [{"lag": lag, "F": t[0]["ssr_ftest"][0], "p_value": t[0]["ssr_ftest"][1]} for lag, t in tests.items()]
    df = pd.DataFrame(rows)
    df["conclusion"] = df.p_value.map(lambda p: verdict(p, f"{cause.name} does not Granger-cause {effect.name}"))
    return df


def decompose(s: pd.Series, period: int) -> pd.DataFrame:
    s = s.dropna()
    res = STL(s.to_numpy(), period=period, robust=True).fit()
    return pd.DataFrame({"observed": s, "trend": res.trend, "seasonal": res.seasonal, "resid": res.resid},
                        index=s.index)


def monthly_table(r: pd.Series) -> pd.DataFrame:
    """Compounded return per calendar month, years as rows."""
    monthly = (1 + r.dropna()).resample("ME").prod() - 1
    return monthly.groupby([monthly.index.year, monthly.index.month]).first().unstack().rename_axis(
        index="year", columns="month")


def seasonality(r: pd.Series, by: str = "month") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Average return per month or weekday, plus ANOVA/Kruskal tests that all groups share a mean."""
    r = r.dropna()
    keys = r.index.month if by == "month" else r.index.day_name().str[:3]
    grouped = r.groupby(keys)
    table = grouped.agg(mean="mean", median="median", std="std", pct_positive=lambda x: (x > 0).mean(), n="count")
    samples = [g.to_numpy() for _, g in grouped if len(g) > 1]
    f, kw = st.f_oneway(*samples), st.kruskal(*samples)
    h0 = f"all {by}s have the same average return"
    tests = pd.DataFrame([result("ANOVA", f.statistic, f.pvalue, h0), result("Kruskal–Wallis", kw.statistic, kw.pvalue, h0)])
    return table, tests
