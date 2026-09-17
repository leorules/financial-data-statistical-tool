import numpy as np
import pandas as pd

ALPHA = 0.05


def periods_per_year(index: pd.Index) -> int:
    """Infer the annualisation factor (252/52/12) from a DatetimeIndex's typical spacing."""
    if len(index) < 3:
        return 252
    days = np.median(np.diff(index.values).astype("timedelta64[D]").astype(float))
    return 252 if days <= 4 else 52 if days <= 10 else 12


def verdict(p: float, h0: str, alpha: float = ALPHA) -> str:
    if not np.isfinite(p):
        return "p-value unavailable"
    action = "reject" if p < alpha else "fail to reject"
    return f"p = {p:.3g} → {action} H0 ({h0}) at {alpha:.0%}"


def result(test: str, statistic: float, p: float, h0: str, **extra) -> pd.Series:
    return pd.Series({"test": test, "statistic": statistic, "p_value": p, **extra, "conclusion": verdict(p, h0)})


def pair(a: pd.Series, b: pd.Series) -> pd.DataFrame:
    """Align two series on common, non-missing dates."""
    return pd.concat([a, b], axis=1, keys=["a", "b"]).dropna()
