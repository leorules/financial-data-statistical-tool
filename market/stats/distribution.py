import numpy as np
import pandas as pd
from scipy import stats as st
from statsmodels.stats.diagnostic import lilliefors

from market.stats.core import result

LEVELS = (0.95, 0.99)


def kde(s: pd.Series, points: int = 200) -> pd.DataFrame:
    s = s.dropna()
    x = np.linspace(s.min(), s.max(), points)
    return pd.DataFrame({"x": x, "kde": st.gaussian_kde(s)(x), "normal": st.norm.pdf(x, s.mean(), s.std())})


def qq(s: pd.Series, dist: str = "norm") -> pd.DataFrame:
    (theoretical, sample), (slope, intercept, _) = st.probplot(s.dropna(), dist=dist)
    return pd.DataFrame({"theoretical": theoretical, "sample": sample, "fit": slope * theoretical + intercept})


def normality(s: pd.Series) -> pd.DataFrame:
    s = s.dropna()
    h0 = "data is normal"
    jb = st.jarque_bera(s)
    sw = st.shapiro(s.sample(5000, random_state=0) if len(s) > 5000 else s)
    ks_stat, ks_p = lilliefors(s, dist="norm")
    ad = st.anderson(s, dist="norm", method="interpolate")
    return pd.DataFrame([
        result("Jarque–Bera", jb.statistic, jb.pvalue, h0),
        result("Shapiro–Wilk", sw.statistic, sw.pvalue, h0),
        result("Kolmogorov–Smirnov (Lilliefors)", ks_stat, ks_p, h0),
        result("Anderson–Darling (p interpolated, 0.01–0.15)", ad.statistic, ad.pvalue, h0),
    ])


def fit(s: pd.Series) -> pd.DataFrame:
    """Fit normal and Student-t; lower AIC is the better fit."""
    s = s.dropna()
    rows = []
    for name, dist in (("normal", st.norm), ("student_t", st.t)):
        params = dist.fit(s)
        loglik = dist.logpdf(s, *params).sum()
        rows.append({"distribution": name, "params": np.round(params, 6).tolist(),
                     "log_likelihood": loglik, "aic": 2 * len(params) - 2 * loglik})
    return pd.DataFrame(rows).sort_values("aic", ignore_index=True)


def var(s: pd.Series, levels: tuple = LEVELS) -> pd.DataFrame:
    """Value at Risk and Conditional VaR as return thresholds (negative = loss)."""
    s = s.dropna()
    mu, sigma, skew, kurt = s.mean(), s.std(), s.skew(), s.kurt()
    rows = []
    for level in levels:
        z = st.norm.ppf(1 - level)
        hist = s.quantile(1 - level)
        z_cf = z + (z**2 - 1) * skew / 6 + (z**3 - 3 * z) * kurt / 24 - (2 * z**3 - 5 * z) * skew**2 / 36
        rows += [
            {"level": level, "method": "historical", "VaR": hist, "CVaR": s[s <= hist].mean()},
            {"level": level, "method": "parametric", "VaR": mu + sigma * z,
             "CVaR": mu - sigma * st.norm.pdf(z) / (1 - level)},
            {"level": level, "method": "cornish_fisher", "VaR": mu + sigma * z_cf, "CVaR": np.nan},
        ]
    return pd.DataFrame(rows)


