import numpy as np
import pandas as pd
from scipy import stats as st
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform

from market.stats.core import periods_per_year

TESTS = {"pearson": st.pearsonr, "spearman": st.spearmanr, "kendall": st.kendalltau}


def matrix(df: pd.DataFrame, method: str = "pearson") -> pd.DataFrame:
    return df.corr(method=method)


def pvalues(df: pd.DataFrame, method: str = "pearson") -> pd.DataFrame:
    cols = df.columns
    out = pd.DataFrame(0.0, index=cols, columns=cols)
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            ab = df[[a, b]].dropna()
            out.loc[a, b] = out.loc[b, a] = TESTS[method](ab[a], ab[b]).pvalue if len(ab) > 2 else np.nan
    return out


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


def lead_lag(a: pd.Series, b: pd.Series, max_lag: int = 10) -> pd.DataFrame:
    """corr(a_t, b_{t-lag}); a peak at positive lag means b leads a."""
    lags = range(-max_lag, max_lag + 1)
    return pd.DataFrame({"lag": list(lags), "corr": [a.corr(b.shift(lag)) for lag in lags]})
