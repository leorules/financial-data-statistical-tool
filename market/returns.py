import numpy as np
import pandas as pd

from market import resample
from market.config import MIN_OBS


PRICE_FIELDS = {"open", "high", "low", "close", "adj_close"}


def price_matrix(long: pd.DataFrame, field: str = "adj_close") -> pd.DataFrame:
    wide = long.pivot(index="date", columns="ticker", values=field).sort_index()
    wide.columns.name = None
    # Yahoo reports 0 for suspended or untraded microcaps, which would make returns infinite.
    return wide.mask(wide <= 0) if field in PRICE_FIELDS else wide


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


# Regions whose session ends after the Australian close, so their move on date D only reaches the ASX
# on D+1. Asian markets close at or before Sydney, so they are left alone.
LATE_CLOSE = {"United States", "Canada", "Mexico", "Brazil", "Global", "Europe", "United Kingdom", "Germany",
              "France", "Italy", "Spain", "Netherlands", "Switzerland", "India"}


def align_closes(wide: pd.DataFrame, regions: dict[str, str]) -> pd.DataFrame:
    """Pair each date with the information the Australian market actually had.

    Same-day daily returns compare the ASX's reaction to yesterday's Wall Street with today's US
    session, which buries the relationship: ASX against the S&P reads 0.10 same-day and 0.60 once
    the US series is carried forward a day.
    """
    late = [c for c in wide.columns if regions.get(c) in LATE_CLOSE]
    return wide.assign(**{c: wide[c].shift(1) for c in late}) if late else wide


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
