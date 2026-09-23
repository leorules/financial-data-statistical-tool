"""The cash rate that actually applied on each date, for excess-return measures.

USD comes from ^IRX, the 13-week Treasury bill discount yield quoted in percent. AUD comes from
BILL.AX, a bank-bill ETF whose trailing total return tracks the cash rate. Dates outside a source's
history fall back to the flat assumption in config.
"""
import pandas as pd

from market import store
from market.config import RISK_FREE

SOURCES = {"USD": "^IRX", "AUD": "BILL.AX"}
BILL_WINDOW = 63  # trading days: BILL.AX has no yield, so annualise its trailing quarter


def _yield_series(currency: str) -> pd.Series:
    ticker = SOURCES.get(currency)
    if not ticker:
        return pd.Series(dtype=float)
    df = store.prices([ticker])
    if df.empty:
        return pd.Series(dtype=float)
    s = df.set_index("date")["adj_close"].sort_index()
    if ticker == "^IRX":
        return s / 100
    return (s / s.shift(BILL_WINDOW)) ** (252 / BILL_WINDOW) - 1


def annual_rate(index: pd.Index, currency: str = "AUD") -> pd.Series:
    """Annualised cash rate aligned to `index`, filled forward, flat fallback where uncovered."""
    rates = _yield_series(currency)
    if rates.empty:
        return pd.Series(RISK_FREE, index=index)
    aligned = rates.reindex(rates.index.union(index)).ffill().reindex(index)
    return aligned.fillna(RISK_FREE)


def label(index: pd.Index, currency: str = "AUD") -> str:
    """One line naming the source, the average rate applied and any fallback."""
    rates = annual_rate(index, currency)
    source = SOURCES.get(currency)
    covered = _yield_series(currency)
    if covered.empty:
        return f"Cash rate: flat {RISK_FREE:.1%} assumption (no {currency} series loaded)"
    uncovered = (index < covered.index.min()).sum() if len(index) else 0
    gap = f", {uncovered} earlier dates at the flat {RISK_FREE:.0%}" if uncovered else ""
    return f"Cash rate: {source}, averaging {rates.mean():.2%} over this window{gap}"
