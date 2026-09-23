import numpy as np
import pandas as pd

from market.config import RISK_FREE
from market.stats.core import excess, periods_per_year


def zscore(s: pd.Series, window: int) -> pd.Series:
    roll = s.rolling(window)
    return (s - roll.mean()) / roll.std()


def bollinger(s: pd.Series, window: int = 20, k: float = 2.0) -> pd.DataFrame:
    mid, sd = s.rolling(window).mean(), s.rolling(window).std()
    return pd.DataFrame({"value": s, "mid": mid, "upper": mid + k * sd, "lower": mid - k * sd})


def ewma_vol(r: pd.Series, lam: float = 0.94, periods: int | None = None) -> pd.Series:
    """RiskMetrics exponentially weighted volatility, annualised."""
    periods = periods or periods_per_year(r.index)
    return np.sqrt(r.pow(2).ewm(alpha=1 - lam).mean() * periods)


def sharpe(r: pd.Series, window: int, periods: int | None = None, rf: float | pd.Series = RISK_FREE) -> pd.Series:
    periods = periods or periods_per_year(r.index)
    roll = excess(r, rf, periods).rolling(window)
    return roll.mean() / r.rolling(window).std() * np.sqrt(periods)


def moments(s: pd.Series, window: int, periods: int | None = None) -> pd.DataFrame:
    """Rolling mean, std, variance, skew, kurtosis, z-score and annualised volatility."""
    periods = periods or periods_per_year(s.index)
    roll = s.rolling(window)
    return pd.DataFrame({
        "mean": roll.mean(), "std": roll.std(), "variance": roll.var(), "skew": roll.skew(),
        "kurtosis": roll.kurt(), "min": roll.min(), "max": roll.max(), "zscore": zscore(s, window),
        "vol_ann": roll.std() * np.sqrt(periods),
    })


def corr(a: pd.Series, b: pd.Series, window: int) -> pd.Series:
    return a.rolling(window).corr(b)


def cov(a: pd.Series, b: pd.Series, window: int) -> pd.Series:
    return a.rolling(window).cov(b)


def beta(r: pd.Series, bench: pd.Series, window: int) -> pd.Series:
    return r.rolling(window).cov(bench) / bench.rolling(window).var()
