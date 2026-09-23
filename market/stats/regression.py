import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson

from market.stats.core import periods_per_year, verdict


def fit(y: pd.Series, X: pd.Series | pd.DataFrame, hac: bool = False):
    """OLS of y on X (with constant), using common non-missing dates.

    `hac` switches to Newey-West standard errors, which allow for autocorrelated and
    heteroskedastic residuals. Coefficients are identical either way; only inference changes.
    """
    data = pd.concat([y, X], axis=1).dropna()
    model = sm.OLS(data.iloc[:, 0], sm.add_constant(data.iloc[:, 1:]))
    lags = int(len(data) ** 0.25)
    return model.fit(cov_type="HAC", cov_kwds={"maxlags": lags}) if hac else model.fit()


def coefficients(res) -> pd.DataFrame:
    ci = res.conf_int()
    return pd.DataFrame({"coef": res.params, "std_err": res.bse, "t": res.tvalues, "p_value": res.pvalues,
                         "ci_low": ci[0], "ci_high": ci[1]})


def diagnostics(res) -> pd.DataFrame:
    dw = durbin_watson(res.resid)
    bp_p = het_breuschpagan(res.resid, res.model.exog)[1] if res.model.exog.shape[1] > 1 else np.nan
    rows = [
        ("R²", res.rsquared, "share of variance explained"),
        ("Adjusted R²", res.rsquared_adj, ""),
        ("Observations", res.nobs, ""),
        ("Residual std", np.sqrt(res.scale), ""),
        ("F-test p-value", res.f_pvalue, verdict(res.f_pvalue, "all slopes are zero")),
        ("Durbin–Watson", dw, "≈2 no autocorrelation; <1.5 positive; >2.5 negative"),
        ("Breusch–Pagan p-value", bp_p, verdict(bp_p, "residuals are homoskedastic")),
        ("Standard errors", np.nan, "Newey–West (HAC)" if res.cov_type == "HAC" else "ordinary OLS"),
    ]
    return pd.DataFrame(rows, columns=["metric", "value", "interpretation"])


def capm(r: pd.Series, bench: pd.Series, periods: int | None = None, hac: bool = False) -> pd.Series:
    """Alpha (annualised) and beta against a benchmark."""
    res = fit(r, bench, hac)
    periods = periods or periods_per_year(r.index)
    alpha, beta = res.params.iloc[0], res.params.iloc[1]
    return pd.Series({"alpha_ann": alpha * periods, "alpha_p": res.pvalues.iloc[0], "beta": beta,
                      "beta_p": res.pvalues.iloc[1], "r2": res.rsquared, "n": int(res.nobs),
                      "conclusion": f"beta {beta:.2f}; alpha {verdict(res.pvalues.iloc[0], 'alpha = 0')}"})


def trend(prices: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Linear fit of log price on time (years). Returns stats and the fitted price line."""
    p = prices.dropna()
    years = pd.Series((p.index - p.index[0]).days / 365.25, index=p.index, name="years")
    res = fit(np.log(p), years)
    fitted = np.exp(res.fittedvalues)
    stats = pd.Series({"annual_growth": np.exp(res.params.iloc[1]) - 1, "r2": res.rsquared,
                       "slope_p": res.pvalues.iloc[1]})
    return stats, fitted
