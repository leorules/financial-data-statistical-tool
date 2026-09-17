import numpy as np
import pandas as pd
import pytest

from market import indicators, resample, returns
from tests.conftest import make_bars


def test_sma_and_ema():
    s = pd.Series([1.0, 2, 3, 4, 5])
    assert indicators.sma(s, 3).tolist()[2:] == [2.0, 3.0, 4.0]
    assert indicators.ema(s, 3).iloc[-1] == pytest.approx(4.0625)


def test_rsi_bounds_and_extremes():
    up = pd.Series(np.arange(1.0, 40))
    assert indicators.rsi(up).iloc[-1] == pytest.approx(100)
    zigzag = pd.Series([10.0, 11] * 20)
    assert indicators.rsi(zigzag).iloc[-1] == pytest.approx(50, abs=5)


def test_drawdown():
    assert indicators.drawdown(pd.Series([100.0, 120, 90, 130])).tolist() == [0, 0, -0.25, 0]


def test_snapshot_metrics(dates, rng):
    close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, len(dates)))), index=dates)
    long = pd.concat([make_bars("AAA.AX", close), make_bars("^AXJO", close)])
    snap = indicators.snapshot(long).set_index("ticker")
    row = snap.loc["AAA.AX"]
    assert row.close == pytest.approx(close.iloc[-1])
    assert row.ret_5d == pytest.approx(close.iloc[-1] / close.iloc[-6] - 1)
    assert row.sma_20 == pytest.approx(close.iloc[-20:].mean())
    assert row.beta_1y == pytest.approx(1) and row.corr_1y == pytest.approx(1)
    assert set(indicators.METRICS) <= set(snap.columns)


def test_resample_ohlcv():
    idx = pd.bdate_range("2024-01-01", "2024-01-12")
    bars = pd.DataFrame({"open": range(10), "high": range(10, 20), "low": range(10), "close": range(10),
                         "adj_close": range(10), "volume": [1] * 10}, index=idx, dtype=float)
    weekly = resample.ohlcv(bars, "W")
    assert weekly.to_dict("records")[0] == {"open": 0, "high": 14, "low": 0, "close": 4, "adj_close": 4, "volume": 5}
    assert len(weekly) == 2


def test_returns_alignment_and_min_obs(dates):
    prices = pd.DataFrame({"A": np.linspace(100, 200, len(dates)), "B": np.linspace(50, 60, len(dates))}, index=dates)
    prices.loc[dates[5], "B"] = np.nan
    prices["C"] = np.nan
    prices.loc[dates[:10], "C"] = 1.0
    r = returns.compute(prices, min_obs=30)
    assert list(r.columns) == ["A", "B"] and r.attrs["dropped"] == ["C"]
    assert dates[5] not in r.index
    assert r["A"].iloc[0] == pytest.approx(prices["A"].iloc[1] / prices["A"].iloc[0] - 1)
    log = returns.compute(prices, kind="log")
    assert np.exp(log["A"].sum()) == pytest.approx(prices["A"].iloc[-1] / prices["A"].iloc[0])


def test_currency_conversion(dates):
    wide = pd.DataFrame({"X.AX": 10.0, "SPY": 10.0}, index=dates)
    fx = pd.Series(0.5, index=dates)
    usd = returns.to_currency(wide, {"X.AX": "AUD", "SPY": "USD"}, fx, "USD")
    aud = returns.to_currency(wide, {"X.AX": "AUD", "SPY": "USD"}, fx, "AUD")
    assert usd["X.AX"].iloc[0] == 5 and usd["SPY"].iloc[0] == 10
    assert aud["SPY"].iloc[0] == 20 and aud["X.AX"].iloc[0] == 10
