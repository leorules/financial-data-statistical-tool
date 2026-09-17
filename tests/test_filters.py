import pandas as pd
import pytest

from market import filters, presets, universe
from market.config import benchmark_for


@pytest.fixture
def snap():
    return pd.DataFrame({"ticker": ["A", "B", "C"], "close": [10.0, 20, 30], "sma_200": [12.0, 15, 25],
                         "rsi_14": [25.0, 55, 75]})


def test_parse_value():
    assert filters.parse_value("5%") == 0.05 and filters.parse_value(" 3 ") == 3.0
    assert filters.parse_value("sma_200") == "sma_200"


def test_metric_vs_metric_and_between(snap):
    records = [{"left": "close", "op": ">", "right": "sma_200", "right2": None},
               {"left": "rsi_14", "op": "between", "right": "30", "right2": "70"},
               {"left": "rsi_14", "op": None, "right": "", "right2": None}]
    rules = filters.rules_from_records(records)
    assert len(rules) == 2
    assert filters.apply(snap, rules).ticker.tolist() == ["B"]
    assert filters.apply(snap, []).equals(snap)


def test_unknown_metric(snap):
    with pytest.raises(ValueError, match="nope"):
        filters.apply(snap, [filters.Rule("close", ">", "nope")])


def test_universe_filter():
    inst = pd.DataFrame({"ticker": ["BHP.AX", "SPY", "CBA.AX"], "name": ["BHP Group", "SPDR", "Commonwealth Bank"],
                         "universe": ["asx200", "indices", "asx200"], "sector": ["Materials", None, "Financials"],
                         "type": ["equity", "etf", "equity"]})
    assert filters.universe(inst, universe=["asx200"], sector=["Materials"]).ticker.tolist() == ["BHP.AX"]
    assert filters.universe(inst, search="bank").ticker.tolist() == ["CBA.AX"]


def test_presets_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(presets, "PRESETS_PATH", tmp_path / "presets.json")
    presets.save("miners", {"sectors": ["Materials"]})
    assert presets.load_all() == {"miners": {"sectors": ["Materials"]}}
    presets.delete("miners")
    assert presets.load_all() == {}


def test_universe_sort_puts_indices_first():
    inst = pd.DataFrame({"ticker": ["AAA.AX", "SPY", "^GSPC", "^AXJO", "ZZZ"],
                         "universe": ["asx200", "etfs", "indices", "indices", "custom"]})
    assert universe.sort(inst).ticker.tolist() == ["^AXJO", "^GSPC", "SPY", "ZZZ", "AAA.AX"]


def test_benchmark_for():
    assert [benchmark_for(t) for t in ["BHP.AX", "^ATLI", "^AXMJ", "^AEX", "SPY"]] == \
        ["^AXJO", "^AXJO", "^AXJO", "^GSPC", "^GSPC"]


def test_built_in_lists_are_unique_and_classified():
    built_in = pd.concat(universe.BUILT_IN.values())
    assert built_in.ticker.is_unique
    assert set(built_in.asset_class) <= set(universe.ASSET_CLASSES)
    assert built_in.set_index("ticker").loc[["^VIX", "GC=F", "TLT", "^TNX"], "asset_class"].tolist() ==         ["Volatility", "Commodities", "Fixed income", "Rates"]


def test_asx_listed_parses_the_exchange_directory(monkeypatch):
    csv = ('"ASX code","Company name","GICs industry group","Listing date","Market Cap"\n'
           '"BHP","BHP GROUP LIMITED","Materials","1885-01-01",200000000\n'
           '"14D","1414 DEGREES LIMITED","Capital Goods","2018-09-12",40720219\n')
    monkeypatch.setattr(universe.requests, "get", lambda *a, **k: type("R", (), {"text": csv})())
    listed = universe.asx_listed()
    assert listed.ticker.tolist() == ["BHP.AX", "14D.AX"]
    assert listed.name.tolist() == ["Bhp Group Limited", "1414 Degrees Limited"]
    assert listed.sector.tolist() == ["Materials", "Capital Goods"]
