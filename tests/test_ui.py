"""Helpers in market/ui.py that can be called without a Streamlit runtime.

These are the ones carrying real logic rather than layout: which markets close late, what a benchmark
key resolves to, and how a blend is priced.
"""
import pandas as pd
import pytest

from market import benchmarks, ui


@pytest.fixture
def settings():
    return ui.Settings(None, pd.Timestamp("2026-09-01").date(), "D", "simple", "native", ())


# --- market regions -----------------------------------------------------------------------------

def test_regions_maps_tickers_and_groups_to_markets(seeded_db):
    where = ui.regions()
    assert where["^AXJO"] == "Australia" and where["^GSPC"] == "United States"
    # A group maps through whichever series stands behind it, including the first leg of a blend.
    grouped = ui.regions({"Equities": "^GSPC", "Blend": {"SPY": 0.5, "^AXJO": 0.5}})
    assert grouped["Equities"] == "United States" and grouped["Blend"] == "United States"


def test_late_closers_finds_only_markets_that_close_after_the_asx(seeded_db):
    late = ui.late_closers(["^AXJO", "^GSPC", "BHP.AX", "SPY", "REET"])
    assert set(late) == {"^GSPC", "SPY", "REET"}
    assert ui.late_closers(["^AXJO", "BHP.AX", "CBA.AX"]) == [], "a local basket needs no alignment"


# --- benchmark resolution -----------------------------------------------------------------------

def test_benchmark_label_reads_a_key(seeded_db):
    assert ui.benchmark_label("^AXJO") == "^AXJO"
    assert ui.benchmark_label("cpi:AUS") == "Australia CPI"
    assert ui.benchmark_label(f"{benchmarks.CPI_PREFIX}USA") == "United States CPI"


def test_benchmark_returns_from_a_ticker(seeded_db, settings):
    r = ui.benchmark_returns("^AXJO", settings)
    assert r is not None and len(r) > 100
    assert r.abs().max() < 1, "these are returns, not prices"


@pytest.mark.parametrize("freq,least", [("D", 500), ("W", 100), ("M", 24)])
def test_an_inflation_benchmark_yields_returns_at_every_interval(seeded_db, settings, freq, least):
    """Resampling a quarterly series straight to the analysis interval used to leave gaps that
    dropped every observation, so weekly and monthly came back empty."""
    r = ui.benchmark_returns("cpi:AUS", ui.Settings(None, settings.end, freq, "simple", "native", ()))
    assert r is not None and len(r) > least, f"{freq} gave {0 if r is None else len(r)} observations"
    assert r.notna().all() and (r.abs() < 0.1).all()


def test_an_inflation_benchmark_recovers_the_underlying_rate(seeded_db, settings):
    """The seeded Australian CPI climbs at exactly 2.5% a year."""
    r = ui.benchmark_returns("cpi:AUS", settings)
    annualised = (1 + r.mean()) ** 252 - 1
    assert annualised == pytest.approx(0.025, abs=0.003)


def test_an_unknown_region_returns_nothing_rather_than_raising(seeded_db, settings):
    assert ui.benchmark_returns("cpi:JPN", settings) is None, "no Japanese rows are seeded"


# --- price matrix -------------------------------------------------------------------------------

def test_price_matrix_resolves_a_blend_into_one_column(seeded_db, settings):
    groups = {"Australia": "^AXJO", "Half and half": {"^AXJO": 0.5, "^GSPC": 0.5}}
    wide = ui.price_matrix(["^AXJO", "^GSPC"], settings, groups=groups)
    assert list(wide.columns) == ["Australia", "Half and half"]
    assert wide["Half and half"].iloc[0] == pytest.approx(100.0), "a blend is indexed from 100"
    assert wide["Half and half"].notna().all()


def test_price_matrix_drops_a_group_whose_series_are_missing(seeded_db, settings):
    wide = ui.price_matrix(["^AXJO"], settings, groups={"Real": "^AXJO", "Missing": "NOT.LOADED",
                                                        "Partial blend": {"^AXJO": 0.5, "NOT.LOADED": 0.5}})
    assert list(wide.columns) == ["Real"]


def test_sources_held_accepts_a_ticker_or_a_complete_blend():
    held = {"AAA", "BBB"}
    assert ui._sources_held("AAA", held)
    assert not ui._sources_held("CCC", held)
    assert ui._sources_held({"AAA": 0.5, "BBB": 0.5}, held)
    assert not ui._sources_held({"AAA": 0.5, "CCC": 0.5}, held), "every leg must be present"
    assert not ui._sources_held(None, held)


# --- cached reads -------------------------------------------------------------------------------

def test_snapshot_filters_the_universe_wide_computation(seeded_db, settings):
    """Metrics are computed once for everything and sliced, so a filtered selection must match."""
    everything = ui.snapshot(tuple(seeded_db.instruments().ticker), settings.end)
    subset = ui.snapshot(("^AXJO", "BHP.AX"), settings.end)
    assert set(subset.ticker) == {"^AXJO", "BHP.AX"}
    merged = everything.set_index("ticker").loc[subset.ticker]
    assert subset.set_index("ticker").close.round(6).equals(merged.close.round(6))


def test_freshness_reports_the_last_run(seeded_db):
    assert not ui.freshness("^AXJO").empty
    assert ui.freshness("NOT.LOADED").empty
