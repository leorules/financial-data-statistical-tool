import numpy as np
import pandas as pd

from market.stats.core import periods_per_year


def longest_streak(mask: pd.Series) -> int:
    runs = mask.groupby((mask != mask.shift()).cumsum()).sum()
    return int(runs.max()) if len(runs) else 0


def binned_mode(s: pd.Series, bins: int = 50) -> float:
    counts, edges = np.histogram(s, bins=bins)
    i = counts.argmax()
    return (edges[i] + edges[i + 1]) / 2


def describe(s: pd.Series, returns: bool = True, log: bool = False, periods: int | None = None) -> pd.Series:
    """Summary statistics for one series; return-specific metrics when returns=True."""
    s = s.dropna()
    if s.empty:
        return pd.Series(dtype=float)
    q1, q3 = s.quantile([0.25, 0.75])
    mean, std = s.mean(), s.std()
    z = (s - mean) / std
    out = {
        "count": len(s), "mean": mean, "median": s.median(), "mode": binned_mode(s), "variance": s.var(),
        "std": std, "sem": s.sem(), "cv": std / abs(mean) if mean else np.nan,
        "min": s.min(), "p5": s.quantile(0.05), "q1": q1, "q3": q3, "p95": s.quantile(0.95), "max": s.max(),
        "range": s.max() - s.min(), "iqr": q3 - q1, "skew": s.skew(), "kurtosis": s.kurt(),
        "last": s.iloc[-1], "last_z": z.iloc[-1],
        "outliers_z3": int((z.abs() > 3).sum()),
        "outliers_iqr": int(((s < q1 - 1.5 * (q3 - q1)) | (s > q3 + 1.5 * (q3 - q1))).sum()),
    }
    if returns:
        periods = periods or periods_per_year(s.index)
        growth = np.exp(s.sum()) if log else (1 + s).prod()
        out |= {
            "cum_return": growth - 1, "geo_mean": growth ** (1 / len(s)) - 1,
            "cagr": growth ** (periods / len(s)) - 1,
            "pct_positive": (s > 0).mean(), "pct_negative": (s < 0).mean(),
            "best": s.max(), "worst": s.min(),
            "win_streak": longest_streak(s > 0), "loss_streak": longest_streak(s < 0),
        }
    return pd.Series(out, name=s.name)


def summary(df: pd.DataFrame, returns: bool = True, log: bool = False) -> pd.DataFrame:
    """describe() for every column; rows are tickers."""
    return pd.DataFrame({c: describe(df[c], returns, log) for c in df.columns}).T


def outliers(s: pd.Series, z: float = 3.0) -> pd.DataFrame:
    s = s.dropna()
    scores = (s - s.mean()) / s.std()
    q1, q3 = s.quantile([0.25, 0.75])
    iqr_flag = (s < q1 - 1.5 * (q3 - q1)) | (s > q3 + 1.5 * (q3 - q1))
    out = pd.DataFrame({"value": s, "z": scores, "z_outlier": scores.abs() > z, "iqr_outlier": iqr_flag})
    return out[out.z_outlier | out.iqr_outlier]
