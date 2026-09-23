import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.covariance import ledoit_wolf

from market.config import RISK_FREE
from market.stats.core import excess, periods_per_year


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


def covariance(r: pd.DataFrame, shrink: bool = False) -> np.ndarray:
    """Sample covariance, or a Ledoit-Wolf shrunk estimate that stays well-conditioned when the
    number of series approaches the number of observations."""
    x = r.dropna()
    return ledoit_wolf(x)[0] if shrink else x.cov().to_numpy()


def shrinkage_intensity(r: pd.DataFrame) -> float:
    """Weight Ledoit-Wolf puts on the structured target; higher means a noisier sample covariance."""
    return float(ledoit_wolf(r.dropna())[1])


def diversification_ratio(r: pd.DataFrame, weights: np.ndarray, shrink: bool = False) -> float:
    cov = covariance(r, shrink)
    return float(weights @ np.sqrt(np.diag(cov)) / np.sqrt(weights @ cov @ weights))


def min_variance_weights(r: pd.DataFrame, long_only: bool = True, shrink: bool = False) -> pd.Series:
    cov = covariance(r, shrink)
    n = len(cov)
    res = minimize(lambda w: w @ cov @ w, np.full(n, 1 / n), method="SLSQP",
                   bounds=[(0, 1)] * n if long_only else None,
                   constraints={"type": "eq", "fun": lambda w: w.sum() - 1})
    return pd.Series(res.x, index=r.columns)


def portfolios(r: pd.DataFrame, rf: float | pd.Series = RISK_FREE,
               shrink: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Equal-weight vs long-only minimum-variance portfolio stats and weights."""
    x = r.dropna()
    periods = periods_per_year(x.index)
    weights = pd.DataFrame({"equal_weight": np.full(x.shape[1], 1 / x.shape[1]),
                            "min_variance": min_variance_weights(x, shrink=shrink)}, index=x.columns)
    rows = {}
    for name, w in weights.items():
        pr = x @ w
        ret, vol = pr.mean() * periods, pr.std() * np.sqrt(periods)
        rows[name] = {"ann_return": ret, "ann_vol": vol,
                      "sharpe": excess(pr, rf, periods).mean() * periods / vol,
                      "diversification_ratio": diversification_ratio(x, w.to_numpy(), shrink)}
    return pd.DataFrame(rows).T, weights
