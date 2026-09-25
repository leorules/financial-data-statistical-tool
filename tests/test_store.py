"""The data layer: upserts, reads, the health queries and the self-migrating schema."""
import duckdb
import pandas as pd
import pytest

from tests.conftest import SEED, make_bars


def bars(ticker: str, start: str, days: int, close: float = 100.0) -> pd.DataFrame:
    dates = pd.bdate_range(start, periods=days)
    return make_bars(ticker, pd.Series(close, index=dates))


# --- writing -----------------------------------------------------------------------------------

def test_upsert_is_idempotent_and_replaces_on_conflict(temp_db):
    first = bars("AAA", "2024-01-01", 5, close=10.0)
    temp_db.upsert("prices", first)
    temp_db.upsert("prices", first)
    assert len(temp_db.prices(["AAA"])) == 5, "re-inserting the same rows must not duplicate them"

    temp_db.upsert("prices", bars("AAA", "2024-01-01", 5, close=99.0))
    assert temp_db.prices(["AAA"]).close.eq(99.0).all(), "the primary key makes a repeat an update"
    assert len(temp_db.prices(["AAA"])) == 5


def test_upsert_ignores_an_empty_frame(temp_db):
    temp_db.upsert("prices", bars("AAA", "2024-01-01", 3))
    temp_db.upsert("prices", pd.DataFrame())
    assert len(temp_db.prices(["AAA"])) == 3


def test_delete_clears_every_table_holding_the_ticker(temp_db):
    temp_db.upsert("instruments", pd.DataFrame([{"ticker": "AAA", "name": "A", "universe": "custom"},
                                                {"ticker": "BBB", "name": "B", "universe": "custom"}]))
    temp_db.upsert("prices", pd.concat([bars("AAA", "2024-01-01", 3), bars("BBB", "2024-01-01", 3)]))
    temp_db.upsert("ingest_log", pd.DataFrame([{"ticker": "AAA", "status": "ok"}, {"ticker": "BBB", "status": "ok"}]))

    temp_db.delete(["AAA"])
    assert temp_db.prices(["AAA"]).empty and temp_db.instruments().ticker.tolist() == ["BBB"]
    assert temp_db.ingest_log().ticker.tolist() == ["BBB"], "the log must not keep an orphan row"


# --- reading -----------------------------------------------------------------------------------

def test_instruments_filters_by_universe(seeded_db):
    assert len(seeded_db.instruments()) == len(SEED)
    assert set(seeded_db.instruments(["asx200"]).ticker) == {"BHP.AX", "CBA.AX"}
    assert seeded_db.instruments(["nothing-here"]).empty


def test_prices_respects_its_date_bounds(seeded_db):
    everything = seeded_db.prices(["^AXJO"])
    window = seeded_db.prices(["^AXJO"], "2024-01-01", "2024-03-31")
    assert len(window) < len(everything)
    assert window.date.min() >= pd.Timestamp("2024-01-01") and window.date.max() <= pd.Timestamp("2024-03-31")
    assert seeded_db.prices(["^AXJO"], None, "2020-01-01").date.max() <= pd.Timestamp("2020-01-01")


def test_date_ranges_reports_first_and_last_per_ticker(temp_db):
    temp_db.upsert("prices", pd.concat([bars("AAA", "2024-01-01", 10), bars("BBB", "2024-06-03", 4)]))
    ranges = temp_db.date_ranges(["AAA", "BBB"]).set_index("ticker")
    assert pd.Timestamp(ranges.loc["AAA", "first_date"]) == pd.Timestamp("2024-01-01")
    assert pd.Timestamp(ranges.loc["BBB", "first_date"]) == pd.Timestamp("2024-06-03")
    assert pd.Timestamp(ranges.loc["AAA", "last_date"]) > pd.Timestamp(ranges.loc["AAA", "first_date"])
    assert temp_db.date_ranges(["missing"]).empty


def test_stats_counts_what_is_held(seeded_db):
    info = seeded_db.stats()
    assert info["instruments"] == len(SEED) and info["tickers"] == len(SEED)
    assert info["rows"] > 1000 and info["size_mb"] > 0
    assert pd.notna(info["last_run"])


# --- health ------------------------------------------------------------------------------------

def test_coverage_separates_instruments_held_from_those_with_prices(seeded_db):
    seeded_db.upsert("instruments", pd.DataFrame([{"ticker": "NODATA.AX", "name": "No prices",
                                                   "universe": "asx200", "type": "equity"}]))
    cover = seeded_db.coverage().set_index("universe")
    assert cover.loc["asx200", "instruments"] == 3
    assert cover.loc["asx200", "with_data"] == 2, "the instrument with no prices is counted but not covered"
    assert pd.notna(cover.loc["indices", "last_date"])


def test_quality_finds_each_problem_it_claims_to(seeded_db):
    seeded_db.upsert("instruments", pd.DataFrame([{"ticker": "NODATA.AX", "name": "No prices",
                                                   "universe": "custom"}]))
    seeded_db.upsert("prices", bars("STALE.AX", "2019-01-01", 5))
    seeded_db.upsert("prices", bars("ZERO.AX", "2026-08-03", 5, close=0.0))
    seeded_db.upsert("ingest_log", pd.DataFrame([{"ticker": "BROKEN.AX", "status": "error", "error": "delisted"}]))

    issues = seeded_db.quality().set_index("issue")
    assert "NODATA.AX" in issues.loc["No price data", "examples"]
    assert "STALE.AX" in issues.loc["Stale by over 7 days", "examples"]
    assert "ZERO.AX" in issues.loc["Zero or negative prices", "examples"]
    assert "BROKEN.AX" in issues.loc["Failed on last refresh", "examples"]
    assert (issues.tickers > 0).all()


def test_a_clean_database_reports_no_problems(temp_db):
    temp_db.upsert("instruments", pd.DataFrame([{"ticker": "AAA", "name": "A", "universe": "custom"}]))
    temp_db.upsert("prices", bars("AAA", "2024-01-01", 5))
    temp_db.upsert("ingest_log", pd.DataFrame([{"ticker": "AAA", "status": "ok"}]))
    issues = temp_db.quality().set_index("issue")
    assert issues.loc["No price data", "tickers"] == 0
    assert issues.loc["Zero or negative prices", "tickers"] == 0


# --- schema ------------------------------------------------------------------------------------

def test_the_schema_migrates_an_older_database_without_losing_rows(tmp_path, monkeypatch):
    """The objective columns were added after portfolios existed, so opening an older file must add
    them rather than fail or start over."""
    from market import store

    path = tmp_path / "old.duckdb"
    with duckdb.connect(str(path)) as con:
        con.execute("CREATE TABLE portfolios(name VARCHAR PRIMARY KEY, cash DOUBLE, benchmark VARCHAR)")
        con.execute("INSERT INTO portfolios VALUES ('legacy', 1000.0, '^AXJO')")

    monkeypatch.setattr(store, "DATA", tmp_path)
    monkeypatch.setattr(store, "DB_PATH", path)
    store._schema_ready.discard(str(path))

    row = store.query("SELECT * FROM portfolios").iloc[0]
    assert row["name"] == "legacy" and row.cash == 1000.0, "the existing row survives"
    assert {"objective_margin", "objective_years", "cpi_region"} <= set(store.query("SELECT * FROM portfolios"))
    assert pd.isna(row.get("objective_margin", pd.NA)) or row.get("objective_margin") is None


def test_schema_readiness_is_tracked_per_file(tmp_path, monkeypatch):
    """A single flag would mean the second database in a test run never gets its tables."""
    from market import store

    for name in ("one.duckdb", "two.duckdb"):
        monkeypatch.setattr(store, "DATA", tmp_path)
        monkeypatch.setattr(store, "DB_PATH", tmp_path / name)
        store.upsert("instruments", pd.DataFrame([{"ticker": "AAA", "name": "A", "universe": "custom"}]))
        assert store.instruments().ticker.tolist() == ["AAA"], name
    assert len({p for p in store._schema_ready if str(tmp_path) in p}) == 2


def test_query_accepts_parameters(seeded_db):
    result = seeded_db.query("SELECT count(*) AS n FROM prices WHERE ticker = ?", ["^AXJO"])
    assert result.n.iloc[0] > 0
    assert seeded_db.query("SELECT count(*) AS n FROM prices WHERE ticker = ?", ["nope"]).n.iloc[0] == 0


@pytest.mark.parametrize("table", ["instruments", "prices", "portfolios", "holdings", "inflation", "ingest_log",
                                   "profiles"])
def test_every_table_exists_on_a_fresh_database(temp_db, table):
    assert temp_db.query(f"SELECT count(*) AS n FROM {table}").n.iloc[0] == 0
