import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def dates():
    return pd.bdate_range("2020-01-01", periods=1000)


@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    from market import store
    monkeypatch.setattr(store, "DATA", tmp_path)
    monkeypatch.setattr(store, "DB_PATH", tmp_path / "test.duckdb")
    return store


def make_bars(ticker: str, close: pd.Series) -> pd.DataFrame:
    return pd.DataFrame({"ticker": ticker, "date": close.index, "open": close.values, "high": close.values * 1.01,
                         "low": close.values * 0.99, "close": close.values, "adj_close": close.values,
                         "volume": 1000.0})


# A ticker per role the pages need: benchmarks, a cash series for the live rate, factor legs,
# equities to screen, and an FX pair for currency conversion.
SEED = [
    ("^AXJO", "S&P/ASX 200", "index", "Australia", "AUD", None, "indices", "Equities"),
    ("^GSPC", "S&P 500", "index", "United States", "USD", None, "indices", "Equities"),
    ("^AXPJ", "ASX 200 A-REIT", "sector", "Australia", "AUD", None, "indices", "Real estate"),
    ("STW.AX", "SPDR S&P/ASX 200 Fund", "etf", "Australia", "AUD", None, "etfs", "Equities"),
    ("VAS.AX", "Vanguard Australian Shares ETF", "etf", "Australia", "AUD", None, "etfs", "Equities"),
    ("SPY", "SPDR S&P 500 ETF", "etf", "United States", "USD", None, "etfs", "Equities"),
    ("IAF.AX", "iShares Core Composite Bond ETF", "etf", "Australia", "AUD", None, "etfs", "Fixed income"),
    ("BILL.AX", "iShares Core Cash ETF", "etf", "Australia", "AUD", None, "etfs", "Cash"),
    ("IFRA.AX", "VanEck FTSE Global Infrastructure ETF", "etf", "Australia", "AUD", None, "etfs", "Infrastructure"),
    ("REET", "iShares Global REIT ETF", "etf", "Global", "USD", None, "etfs", "Real estate"),
    ("DJP", "iPath Bloomberg Commodity ETN", "etn", "Global", "USD", None, "etfs", "Commodities"),
    ("TLT", "iShares 20+ Year Treasury Bond ETF", "etf", "United States", "USD", None, "etfs", "Fixed income"),
    ("SHY", "iShares 1-3 Year Treasury Bond ETF", "etf", "United States", "USD", None, "etfs", "Fixed income"),
    ("BHP.AX", "BHP Group", "equity", "Australia", "AUD", "Materials", "asx200", "Equities"),
    ("CBA.AX", "Commonwealth Bank", "equity", "Australia", "AUD", "Financials", "asx200", "Equities"),
    ("^IRX", "US 13-Week T-Bill Yield", "yield", "United States", "USD", None, "rates", "Rates"),
    ("AUDUSD=X", "AUD/USD", "fx", "Australia", "AUD", None, "fx", "Currencies"),
]


@pytest.fixture
def seeded_db(tmp_path, monkeypatch):
    """A complete miniature database, isolated from the real one.

    Every path below is a module-level constant bound at import, so patching the database alone would
    leave stress periods and presets writing into the real data directory.
    """
    import streamlit as st

    from market import presets, store, stress, universe

    monkeypatch.setattr(store, "DATA", tmp_path)
    monkeypatch.setattr(store, "DB_PATH", tmp_path / "test.duckdb")
    monkeypatch.setattr(stress, "PERIODS_PATH", tmp_path / "stress_periods.json")
    monkeypatch.setattr(stress, "LEGACY_PATH", tmp_path / "events.json")
    monkeypatch.setattr(presets, "PRESETS_PATH", tmp_path / "presets.json")
    monkeypatch.setattr(universe, "UNIVERSE_DIR", tmp_path / "universe")
    # Eight cached loaders in market.ui would otherwise serve the real database.
    st.cache_data.clear()
    st.cache_resource.clear()

    dates = pd.bdate_range("2019-01-01", "2026-09-01")
    rng = np.random.default_rng(0)
    instruments = pd.DataFrame(SEED, columns=["ticker", "name", "type", "exchange", "currency", "sector",
                                              "universe", "asset_class"])
    store.upsert("instruments", instruments)

    drift = {"^IRX": 0.0, "BILL.AX": 0.00002, "AUDUSD=X": 0.0}
    bars = [make_bars(t, pd.Series(100 * np.cumprod(1 + rng.normal(drift.get(t, 0.0003), 0.01, len(dates))),
                                   index=dates)) for t in instruments.ticker]
    store.upsert("prices", pd.concat(bars, ignore_index=True))

    quarters = pd.period_range("2000Q1", periods=107, freq="Q").to_timestamp(how="end").normalize()
    months = pd.period_range("2000-01", periods=320, freq="M").to_timestamp(how="end").normalize()
    store.upsert("inflation", pd.concat([
        pd.DataFrame({"region": "AUS", "date": quarters, "cpi": 100 * 1.025 ** (np.arange(len(quarters)) / 4)}),
        pd.DataFrame({"region": "USA", "date": months, "cpi": 100 * 1.022 ** (np.arange(len(months)) / 12)})]))

    store.upsert("ingest_log", pd.DataFrame({
        "ticker": instruments.ticker, "first_date": dates[0], "last_date": dates[-1],
        "last_run": pd.Timestamp.now(), "status": "ok", "error": None}))

    from market import portfolio
    portfolio.save("Test portfolio", pd.DataFrame({"ticker": ["BHP.AX", "CBA.AX"], "units": [500.0, 200.0],
                                                   "cost_price": [30.0, 70.0]}), 10_000.0, "^AXJO")
    yield store
    st.cache_data.clear()
    st.cache_resource.clear()
