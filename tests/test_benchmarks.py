import pandas as pd

from market import benchmarks, universe


def test_every_benchmark_is_a_built_in_instrument():
    built_in = pd.concat(universe.BUILT_IN.values()).set_index("ticker")
    priced = benchmarks.TABLE.dropna(subset=["ticker"])
    assert priced.ticker.isin(built_in.index).all()
    assert benchmarks.TABLE.asset_class.isin(universe.ASSET_CLASSES).all()
    assert (benchmarks.TABLE.groupby(["basis", "asset_class"]).headline.sum() == 1).all()


def test_select_headline_and_regional():
    headline = benchmarks.select(["Equities", "Fixed income"])
    assert headline.label.tolist() == ["Equities", "Fixed income"]
    assert headline.ticker.tolist() == ["^892400-USD-STRD", "AGGG.L"]
    regional = benchmarks.select(["Equities", "Cash"], ["Australia"])
    assert regional.label.tolist() == ["Equities · Australia", "Cash · Australia"]
    assert regional.ticker.tolist() == ["^AXJO", "BILL.AX"]


def test_the_two_benchmark_categories_are_separate_and_complete():
    assert benchmarks.BASES == ["Standard", "APRA"]
    standard, apra = (benchmarks.TABLE[benchmarks.TABLE.basis == b] for b in benchmarks.BASES)
    assert len(apra) == 26, "APRA prescribes 26 covered asset classes"
    assert set(standard.region) <= set(benchmarks.REGIONS["Standard"])
    assert set(apra.region) <= set(benchmarks.REGIONS["APRA"])
    # Every prescribed row is priced from a ticker, built as a blend, or explicitly unavailable with a reason.
    assert (apra.ticker.notna() == (apra.series == "proxy")).all()
    assert (apra.series.isin(["proxy", "composite", "unavailable"])).all()
    assert apra[apra.series == "unavailable"].note.str.len().gt(0).all()


def test_selecting_by_basis_returns_only_that_category():
    for basis in benchmarks.BASES:
        rows = benchmarks.select(benchmarks.classes(basis), benchmarks.REGIONS[basis], basis)
        assert (rows.basis == basis).all() and len(rows)
        assert rows.label.is_unique


def test_apra_codes_are_recorded_for_published_indices():
    apra = benchmarks.TABLE[benchmarks.TABLE.basis == "APRA"]
    assert apra.set_index("benchmark").code.get("S&P/ASX 300 Total Return") == "ASA52"
    assert apra[apra.code.notna()].shape[0] == 19, "four unlisted indices and three composites have no code"


def test_the_alternatives_composites_are_built_from_the_prescribed_rows():
    apra = benchmarks.TABLE[benchmarks.TABLE.basis == "APRA"]
    composites = apra[apra.series == "composite"]
    assert len(composites) == 3, "APRA defines three alternatives blends"
    for _, row in composites.iterrows():
        assert abs(sum(row.blend.values()) - 1) < 1e-9
        assert set(row.blend) <= set(apra.ticker.dropna()), "components must be other prescribed rows"
        assert benchmarks.source(row) == row.blend
    growth = composites.set_index("variant").loc["growth", "blend"]
    assert growth == {"VGAD.AX": 0.375, "VGS.AX": 0.375, "VBND.AX": 0.25}


def test_source_prefers_a_blend_then_a_ticker():
    equities = benchmarks.select(["Equities"], ["Australia"], "APRA").iloc[0]
    assert benchmarks.source(equities) == "VAS.AX"
    assert "via 37.5% VGAD.AX" in benchmarks.describe(
        benchmarks.TABLE[benchmarks.TABLE.variant == "growth"].iloc[0])


def test_a_single_company_is_never_offered_as_a_benchmark():
    inst = pd.DataFrame({
        "ticker": ["^AXJO", "STW.AX", "BHP.AX", "^AXPJ", "DJP"],
        "name": ["S&P/ASX 200", "SPDR ASX 200", "BHP Group", "ASX 200 A-REIT", "Bloomberg Commodity ETN"],
        "type": ["index", "etf", "equity", "sector", "etn"]})
    options = benchmarks.eligible(inst)
    assert "BHP.AX" not in set(options.key), "a stock is a holding, not a yardstick"
    assert {"^AXJO", "STW.AX", "^AXPJ", "DJP"} <= set(options.key)
    # Inflation is offered for every hub, and the official benchmarks come first.
    assert options.key.str.startswith(benchmarks.CPI_PREFIX).sum() >= 5
    assert options.group.iloc[0] == "Asset-class benchmark"
    assert options.key.is_unique


def test_inflation_keys_are_recognised():
    assert benchmarks.is_inflation("cpi:AUS")
    assert not benchmarks.is_inflation("^AXJO")
    assert not benchmarks.is_inflation(None)
