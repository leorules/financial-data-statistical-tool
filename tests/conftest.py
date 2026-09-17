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
