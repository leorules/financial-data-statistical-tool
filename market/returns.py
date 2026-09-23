import numpy as np
import pandas as pd

from market import resample
from market.config import MIN_OBS


def price_matrix(long: pd.DataFrame, field: str = "adj_close") -> pd.DataFrame:
    wide = long.pivot(index="date", columns="ticker", values=field).sort_index()
    wide.columns.name = None
    return wide


def to_currency(wide: pd.DataFrame, currencies: dict[str, str], audusd: pd.Series, target: str) -> pd.DataFrame:
    """Convert AUD/USD priced columns into the target currency using the AUDUSD rate."""
    fx = audusd.reindex(wide.index).ffill()
    out = wide.copy()
    for col in wide.columns:
        ccy = currencies.get(col, target)
        if ccy == "AUD" and target == "USD":
            out[col] = wide[col] * fx
        elif ccy == "USD" and target == "AUD" and col != "AUDUSD=X":
            out[col] = wide[col] / fx
    return out


def blend(wide: pd.DataFrame, weights: dict[str, float]) -> pd.Series:
    """Fixed-weight composite index from 100: the weighted average of component returns, compounded.

    Constant weights mean the composite is rebalanced every period, which is how a blended benchmark
    is defined. Periods where any component is missing are dropped.
    """
    parts = wide[list(weights)].dropna()
    if parts.empty:
        return pd.Series(dtype=float)
    w = pd.Series(weights, dtype=float)
    steps = parts.pct_change(fill_method=None).fillna(0).mul(w / w.sum(), axis=1).sum(axis=1)
    return (1 + steps).cumprod() * 100


def compute(prices: pd.DataFrame, freq: str = "D", kind: str = "simple",
            align: str = "inner", min_obs: int = MIN_OBS) -> pd.DataFrame:
    """Aligned return matrix. `inner` keeps only dates where every ticker traded; `ffill` carries prices over gaps."""
    p = resample.last(prices, freq)
    enough = p.count() > min_obs
    p = p.loc[:, enough]
    p = p.dropna() if align == "inner" else p.ffill(limit=5).dropna(how="all")
    r = np.log(p).diff() if kind == "log" else p.pct_change(fill_method=None)
    r = r.iloc[1:]
    r.attrs["dropped"] = enough[~enough].index.tolist()
    return r
