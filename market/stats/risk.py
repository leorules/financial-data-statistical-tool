import numpy as np
import pandas as pd

from market.config import RISK_FREE
from market.stats.core import excess, pair, periods_per_year


def wealth(r: pd.Series) -> pd.Series:
    return (1 + r.fillna(0)).cumprod()


def drawdown(r: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    w = (1 + r.fillna(0)).cumprod()
    return w / w.cummax() - 1


def ann_return(r: pd.Series, periods: int) -> float:
    r = r.dropna()
    return (1 + r).prod() ** (periods / len(r)) - 1


def ann_vol(r: pd.Series, periods: int) -> float:
    return r.std() * np.sqrt(periods)


def downside_dev(r: pd.Series, periods: int, mar: float = 0.0) -> float:
    return np.sqrt((np.minimum(r.dropna() - mar, 0) ** 2).mean() * periods)


def max_drawdown(r: pd.Series) -> dict:
    """Depth, peak/trough/recovery dates and length (periods from peak to recovery or end)."""
    dd = drawdown(r)
    trough = dd.idxmin()
    peak = wealth(r)[:trough].idxmax()
    recovered = dd[trough:][dd[trough:] >= 0]
    recovery = recovered.index[0] if len(recovered) else pd.NaT
    end = recovery if len(recovered) else dd.index[-1]
    return {"max_drawdown": dd.min(), "peak": peak, "trough": trough, "recovery": recovery,
            "dd_length": len(dd[peak:end]) - 1}


def ulcer_index(r: pd.Series) -> float:
    return np.sqrt((drawdown(r) ** 2).mean())


def beta(r: pd.Series, bench: pd.Series) -> float:
    ab = pair(r, bench)
    return ab.cov().iloc[0, 1] / ab["b"].var()


def capture(r: pd.Series, bench: pd.Series) -> tuple[float, float]:
    ab = pair(r, bench)
    up, down = ab[ab.b > 0], ab[ab.b < 0]
    return up.a.mean() / up.b.mean(), down.a.mean() / down.b.mean()


def metrics(r: pd.Series, bench: pd.Series | None = None, periods: int | None = None,
            rf: float | pd.Series = RISK_FREE) -> pd.Series:
    r = r.dropna()
    periods = periods or periods_per_year(r.index)
    mean_excess = excess(r, rf, periods).mean() * periods
    vol, down = ann_vol(r, periods), downside_dev(r, periods)
    mdd = max_drawdown(r)
    out = {
        "ann_return": ann_return(r, periods), "ann_vol": vol, "downside_dev": down,
        "sharpe": mean_excess / vol, "sortino": mean_excess / down,
        "calmar": ann_return(r, periods) / abs(mdd["max_drawdown"]) if mdd["max_drawdown"] else np.nan,
        **mdd, "ulcer_index": ulcer_index(r),
    }
    if bench is not None:
        ab = pair(r, bench)
        active = ab.a - ab.b
        b = beta(r, bench)
        te = active.std() * np.sqrt(periods)
        up, dn = capture(r, bench)
        out |= {"beta": b, "treynor": mean_excess / b, "tracking_error": te,
                "information_ratio": active.mean() * periods / te if te else np.nan, "up_capture": up, "down_capture": dn}
    return pd.Series(out, name=r.name)


def summary(df: pd.DataFrame, bench: pd.Series | None = None, periods: int | None = None,
            rf: float | pd.Series = RISK_FREE) -> pd.DataFrame:
    """Risk and performance metrics for every column; rows are tickers."""
    periods = periods or periods_per_year(df.index)
    return pd.DataFrame({c: metrics(df[c], bench, periods, rf) for c in df.columns}).T


def cross_section(r: pd.DataFrame, periods: int | None = None, rf: float | pd.Series = RISK_FREE) -> pd.DataFrame:
    """Key risk metrics for many series at once; each column uses only its own dates."""
    periods = periods or periods_per_year(r.index)
    var95 = r.quantile(0.05)
    std = r.std()
    vol = std * np.sqrt(periods)
    downside = np.sqrt((r.clip(upper=0) ** 2).mean() * periods)
    mean_excess = excess(r, rf, periods).mean() * periods
    return pd.DataFrame({
        "observations": r.count(), "std": std, "vol_ann": vol, "downside_dev": downside,
        "var_95": var95, "cvar_95": r.where(r.le(var95)).mean(), "var_99": r.quantile(0.01),
        "worst": r.min(), "max_drawdown": drawdown(r).min(), "sharpe": mean_excess / vol, "sortino": mean_excess / downside,
    })
