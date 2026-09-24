"""Risk factors built from listed proxies, for decomposing a return into what drove it.

Each factor is one series, or a long-short spread where the interesting part is the difference:
small companies against large, high yield against investment grade, long bonds against short.
Spreads are used because the level of either leg is mostly market direction, not the factor.
"""
import pandas as pd

# name: (long leg, short leg or None, what it measures)
FACTORS = {
    "Market (US)": ("SPY", None, "Broad US equity market"),
    "Market (AU)": ("VAS.AX", None, "Broad Australian equity market"),
    "Size": ("IWM", "SPY", "Small companies minus large"),
    "Duration": ("TLT", "SHY", "Long bonds minus short: sensitivity to rates"),
    "Credit": ("HYG", "LQD", "High yield minus investment grade: appetite for credit risk"),
    "Commodities": ("DJP", None, "Broad commodity futures"),
    "Gold": ("GLD", None, "Gold bullion"),
    "AUD": ("AUDUSD=X", None, "Australian dollar against the US dollar"),
}


def legs(names: list[str]) -> list[str]:
    """Every ticker the chosen factors need."""
    wanted = [t for name in names for t in FACTORS[name][:2] if t]
    return list(dict.fromkeys(wanted))


def returns(prices: pd.DataFrame, names: list[str]) -> pd.DataFrame:
    """Factor return series from a wide price matrix, skipping any whose legs are missing."""
    r = prices.pct_change(fill_method=None)
    built = {}
    for name in names:
        long_leg, short_leg, _ = FACTORS[name]
        if long_leg not in r or (short_leg and short_leg not in r):
            continue
        built[name] = r[long_leg] - r[short_leg] if short_leg else r[long_leg]
    return pd.DataFrame(built).dropna(how="all")


def describe(name: str) -> str:
    long_leg, short_leg, what = FACTORS[name]
    return f"{what} ({long_leg} − {short_leg})" if short_leg else f"{what} ({long_leg})"


def exposures(result, periods: int) -> pd.DataFrame:
    """Factor betas with their significance, and the annualised alpha left unexplained."""
    table = pd.DataFrame({"beta": result.params, "t": result.tvalues, "p_value": result.pvalues})
    table.loc["const", "beta"] *= periods  # alpha is a per-period intercept; show it per year
    return table.rename(index={"const": "Alpha (annual)"})
