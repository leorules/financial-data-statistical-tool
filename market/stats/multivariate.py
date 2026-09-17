import numpy as np
import pandas as pd
from scipy.optimize import minimize

from market.config import RISK_FREE
from market.stats.core import periods_per_year


def pca(r: pd.DataFrame, standardize: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Explained variance per component and loadings (components as columns)."""
    x = r.dropna()
    m = x.corr() if standardize else x.cov()
    values, vectors = np.linalg.eigh(m.to_numpy())
    order = values.argsort()[::-1]
    values, vectors = values[order], vectors[:, order]
    vectors *= np.sign(vectors.sum(axis=0))
    names = [f"PC{i + 1}" for i in range(len(values))]
    ratio = values / values.sum()
    explained = pd.DataFrame({"component": names, "explained": ratio, "cumulative": ratio.cumsum()})
    return explained, pd.DataFrame(vectors, index=r.columns, columns=names)


def diversification_ratio(r: pd.DataFrame, weights: np.ndarray) -> float:
    cov = r.dropna().cov().to_numpy()
    return float(weights @ np.sqrt(np.diag(cov)) / np.sqrt(weights @ cov @ weights))


def min_variance_weights(r: pd.DataFrame, long_only: bool = True) -> pd.Series:
    cov = r.dropna().cov().to_numpy()
    n = len(cov)
    res = minimize(lambda w: w @ cov @ w, np.full(n, 1 / n), method="SLSQP",
                   bounds=[(0, 1)] * n if long_only else None,
                   constraints={"type": "eq", "fun": lambda w: w.sum() - 1})
    return pd.Series(res.x, index=r.columns)


def portfolios(r: pd.DataFrame, rf: float = RISK_FREE) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Equal-weight vs long-only minimum-variance portfolio stats and weights."""
    x = r.dropna()
    periods = periods_per_year(x.index)
    weights = pd.DataFrame({"equal_weight": np.full(x.shape[1], 1 / x.shape[1]),
                            "min_variance": min_variance_weights(x)}, index=x.columns)
    rows = {}
    for name, w in weights.items():
        pr = x @ w
        ret, vol = pr.mean() * periods, pr.std() * np.sqrt(periods)
        rows[name] = {"ann_return": ret, "ann_vol": vol, "sharpe": (ret - rf) / vol,
                      "diversification_ratio": diversification_ratio(x, w.to_numpy())}
    return pd.DataFrame(rows).T, weights
