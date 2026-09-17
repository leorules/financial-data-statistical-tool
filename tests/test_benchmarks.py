import pandas as pd

from market import benchmarks, universe


def test_every_benchmark_is_a_built_in_instrument():
    built_in = pd.concat(universe.BUILT_IN.values()).set_index("ticker")
    assert benchmarks.TABLE.ticker.isin(built_in.index).all()
    assert benchmarks.TABLE.asset_class.isin(universe.ASSET_CLASSES).all()
    assert (benchmarks.TABLE.groupby("asset_class").headline.sum() == 1).all()


def test_select_headline_and_regional():
    headline = benchmarks.select(["Equities", "Fixed income"])
    assert headline.label.tolist() == ["Equities", "Fixed income"]
    assert headline.ticker.tolist() == ["^892400-USD-STRD", "AGGG.L"]
    regional = benchmarks.select(["Equities", "Cash"], ["Australia"])
    assert regional.label.tolist() == ["Equities · Australia", "Cash · Australia"]
    assert regional.ticker.tolist() == ["^AXJO", "BILL.AX"]
