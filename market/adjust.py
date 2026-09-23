"""Optional methodology adjustments. All off by default: the tool reports the data as it is until
the user opts into a correction, and every page says which ones are active."""

ADJUSTMENTS = {
    "total_return": ("Total-return benchmarks",
                     "Measure beta, alpha and capture against an accumulation series (STW.AX, ^SP500TR) "
                     "instead of the price index, which excludes dividends."),
    "live_cash": ("Live cash rate",
                  "Subtract the cash rate that actually applied on each date (^IRX, BILL.AX) instead of "
                  "a flat 4% assumption."),
    "hac": ("Robust standard errors",
            "Use Newey–West errors in regressions, which allow for autocorrelated and "
            "heteroskedastic returns. Coefficients are unchanged; significance moves."),
    "fdr": ("False-discovery correction",
            "Adjust correlation p-values for the number of pairs tested at once, so chance "
            "findings are not read as real."),
    "shrinkage": ("Covariance shrinkage",
                  "Shrink the covariance matrix towards a stable target (Ledoit–Wolf) before "
                  "optimising portfolio weights."),
}


def label(key: str) -> str:
    return ADJUSTMENTS[key][0]


def active(enabled, *keys: str) -> list[str]:
    """Labels of the enabled adjustments among `keys`, in registry order."""
    return [label(k) for k in (keys or ADJUSTMENTS) if k in enabled]
