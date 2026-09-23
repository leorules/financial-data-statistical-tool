import numpy as np
import pandas as pd

from market.config import TOTAL_RETURN, benchmark_for

YEAR = 252
LOOKBACKS = {"ret_1d": 1, "ret_5d": 5, "ret_1m": 21, "ret_3m": 63, "ret_1y": YEAR}


def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n).mean()


def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def rsi(s: pd.Series, n: int = 14) -> pd.Series:
    """Wilder's RSI."""
    delta = s.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    return 100 - 100 / (1 + gain / loss)


def drawdown(s: pd.Series) -> pd.Series:
    return s / s.cummax() - 1


def pct_ago(s: pd.Series, n: int) -> float:
    return s.iloc[-1] / s.iloc[-1 - n] - 1 if len(s) > n and s.iloc[-1 - n] else np.nan


def metrics(bars: pd.DataFrame, bench_returns: pd.DataFrame, total_return: bool = False) -> dict:
    """Snapshot metrics for one ticker's date-indexed daily bars."""
    close, adj = bars["close"], bars["adj_close"]
    year = bars.iloc[-YEAR:]
    ret = adj.pct_change(fill_method=None)
    prev_year = adj[adj.index.year < adj.index[-1].year]
    out = {"date": bars.index[-1], "close": close.iloc[-1]}
    out |= {k: pct_ago(adj, n) for k, n in LOOKBACKS.items()}
    out["ret_ytd"] = adj.iloc[-1] / prev_year.iloc[-1] - 1 if len(prev_year) and prev_year.iloc[-1] else np.nan
    out |= {f"sma_{n}": sma(close, n).iloc[-1] for n in (20, 50, 200)}
    out |= {
        "ema_20": ema(close, 20).iloc[-1],
        "rsi_14": rsi(close).iloc[-1],
        "vol_20d": ret.iloc[-20:].std() * np.sqrt(YEAR),
        "vol_1y": ret.iloc[-YEAR:].std() * np.sqrt(YEAR),
        "max_dd_1y": drawdown(year["adj_close"]).min(),
        "pct_from_52w_high": close.iloc[-1] / year["high"].max() - 1,
        "pct_from_52w_low": close.iloc[-1] / year["low"].min() - 1,
        "volume_ratio_20d": bars["volume"].iloc[-1] / (bars["volume"].iloc[-21:-1].mean() or np.nan),
    }
    bench = benchmark_for(bars["ticker"].iloc[0], total_return)
    if bench in bench_returns:
        pair = pd.concat([ret, bench_returns[bench]], axis=1).iloc[-YEAR:].dropna()
        cov = pair.cov().iloc[0, 1]
        out["beta_1y"] = cov / pair.iloc[:, 1].var() if len(pair) > 20 else np.nan
        out["corr_1y"] = pair.corr().iloc[0, 1] if len(pair) > 20 else np.nan
    return out


BENCHMARKS = ("^AXJO", "^GSPC", *TOTAL_RETURN.values())


def snapshot(long: pd.DataFrame, as_of=None, total_return: bool = False) -> pd.DataFrame:
    """One row of metrics per ticker, as of the given date."""
    df = long if as_of is None else long[long["date"] <= pd.Timestamp(as_of)]
    adj = df.pivot(index="date", columns="ticker", values="adj_close")
    bench_returns = adj[[b for b in BENCHMARKS if b in adj]].pct_change(fill_method=None)
    rows = [{"ticker": t} | metrics(g.set_index("date"), bench_returns, total_return)
            for t, g in df.groupby("ticker") if len(g) > 1]
    return pd.DataFrame(rows)


METRICS = ["close", "ret_1d", "ret_5d", "ret_1m", "ret_3m", "ret_ytd", "ret_1y", "sma_20", "sma_50", "sma_200",
           "ema_20", "rsi_14", "vol_20d", "vol_1y", "max_dd_1y", "pct_from_52w_high", "pct_from_52w_low",
           "volume_ratio_20d", "beta_1y", "corr_1y"]
