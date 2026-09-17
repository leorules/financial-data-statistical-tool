import pandas as pd

RULES = {"W": "W-FRI", "M": "ME"}
OHLCV = {"open": "first", "high": "max", "low": "min", "close": "last", "adj_close": "last", "volume": "sum"}


def ohlcv(bars: pd.DataFrame, freq: str) -> pd.DataFrame:
    """Resample one ticker's date-indexed daily bars to W or M."""
    if freq == "D":
        return bars
    return bars.resample(RULES[freq]).agg(OHLCV).dropna(subset=["close"])


def last(wide: pd.DataFrame, freq: str) -> pd.DataFrame:
    """Resample a wide price matrix by taking each period's last value."""
    return wide if freq == "D" else wide.resample(RULES[freq]).last()


def total(wide: pd.DataFrame, freq: str) -> pd.DataFrame:
    """Resample a wide matrix of flows (e.g. volume) by summing each period."""
    return wide if freq == "D" else wide.resample(RULES[freq]).sum(min_count=1)
