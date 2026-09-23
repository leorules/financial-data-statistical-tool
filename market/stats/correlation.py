import numpy as np
import pandas as pd
from scipy import stats as st
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform

from statsmodels.stats.multitest import multipletests

from market.stats.core import periods_per_year

TESTS = {"pearson": st.pearsonr, "spearman": st.spearmanr, "kendall": st.kendalltau}


def matrix(df: pd.DataFrame, method: str = "pearson") -> pd.DataFrame:
    return df.corr(method=method)


def pvalues(df: pd.DataFrame, method: str = "pearson", adjust: bool = False) -> pd.DataFrame:
    """Pairwise p-values for H0: no correlation.

    Pearson uses its closed form and Spearman scipy's matrix routine. Both are exact, and both avoid
    running a separate test per pair, which grows with the square of the basket. `adjust` applies a
    Benjamini-Hochberg false-discovery correction across every pair tested, so chance findings in a
    large matrix are not read as real.
    """
    if method == "pearson":
        out = _from_t(df.corr(), _overlap(df))
    elif method == "spearman" and df.shape[1] > 2:
        out = _square(st.spearmanr(df.to_numpy(), nan_policy="omit").pvalue, df.columns)
    else:
        out = _per_pair(df, method)
    return correct(out) if adjust else out


def _overlap(df: pd.DataFrame) -> pd.DataFrame:
    """Observations each pair of columns shares."""
    present = df.notna().astype(int)
    return present.T @ present


def _from_t(corr: pd.DataFrame, n: pd.DataFrame) -> pd.DataFrame:
    """Pearson p-values from the correlation matrix: t = r * sqrt((n - 2) / (1 - r^2))."""
    dof = (n - 2).clip(lower=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = corr * np.sqrt(dof / (1 - corr ** 2))
    return _square(2 * st.t.sf(np.abs(t.to_numpy()), dof.to_numpy()), corr.columns)


def _square(values, columns) -> pd.DataFrame:
    p = np.array(values, dtype=float).reshape(len(columns), len(columns))
    np.fill_diagonal(p, 0.0)
    return pd.DataFrame(p, index=columns, columns=columns)


def _per_pair(df: pd.DataFrame, method: str) -> pd.DataFrame:
    """One test per pair, for Kendall and for baskets too small for the matrix routines."""
    cols = df.columns
    out = pd.DataFrame(0.0, index=cols, columns=cols)
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            ab = df[[a, b]].dropna()
            out.loc[a, b] = out.loc[b, a] = TESTS[method](ab[a], ab[b]).pvalue if len(ab) > 2 else np.nan
    return out


def correct(p: pd.DataFrame) -> pd.DataFrame:
    """Benjamini-Hochberg across the upper triangle, mirrored back into a full matrix."""
    rows, cols = np.triu_indices(len(p), k=1)
    raw = p.to_numpy()[rows, cols]
    usable = np.isfinite(raw)
    if not usable.any():
        return p
    adjusted = raw.copy()
    adjusted[usable] = multipletests(raw[usable], method="fdr_bh")[1]
    out = p.to_numpy(copy=True)
    out[rows, cols] = out[cols, rows] = adjusted
    return pd.DataFrame(out, index=p.index, columns=p.columns)


def covariance(df: pd.DataFrame, annualise: bool = True) -> pd.DataFrame:
    return df.cov() * (periods_per_year(df.index) if annualise else 1)


def partial(df: pd.DataFrame) -> pd.DataFrame:
    """Correlation between each pair after controlling for all other columns."""
    precision = np.linalg.pinv(df.dropna().cov().to_numpy())
    d = np.sqrt(np.diag(precision))
    pc = -precision / np.outer(d, d)
    np.fill_diagonal(pc, 1.0)
    return pd.DataFrame(pc, index=df.columns, columns=df.columns)


def cluster_order(corr: pd.DataFrame) -> list[str]:
    if len(corr) < 3:
        return list(corr.columns)
    dist = squareform((1 - corr.fillna(0)).clip(lower=0).to_numpy(), checks=False)
    return list(corr.columns[leaves_list(linkage(dist, "average"))])


def pairs(corr: pd.DataFrame) -> pd.DataFrame:
    """Unique pairs sorted from most to least correlated."""
    upper = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1)).stack().dropna()
    return (upper.rename("corr").rename_axis(["a", "b"]).reset_index()
            .sort_values("corr", ascending=False, ignore_index=True))


def rolling_average(r: pd.DataFrame, window: int) -> pd.Series:
    """Average pairwise correlation through time, which is when diversification holds or fails.

    For a common average correlation p, Var(sum x) = sum(var) + p * ((sum sd)^2 - sum(var)). Solving
    for p needs only a rolling standard deviation per column and one for the basket, so no
    correlation matrix is formed per window.
    """
    sd = r.rolling(window).std()
    own = (sd ** 2).sum(axis=1)
    spread = sd.sum(axis=1) ** 2 - own
    basket = r.sum(axis=1).rolling(window).std() ** 2
    return ((basket - own) / spread).replace([np.inf, -np.inf], np.nan).dropna().clip(-1, 1)


def lead_lag(a: pd.Series, b: pd.Series, max_lag: int = 10) -> pd.DataFrame:
    """corr(a_t, b_{t-lag}); a peak at positive lag means b leads a."""
    lags = range(-max_lag, max_lag + 1)
    return pd.DataFrame({"lag": list(lags), "corr": [a.corr(b.shift(lag)) for lag in lags]})
