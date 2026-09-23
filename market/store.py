from contextlib import contextmanager

import duckdb
import pandas as pd

from market.config import DATA, DB_PATH

SCHEMA = """
CREATE TABLE IF NOT EXISTS instruments(
    ticker VARCHAR PRIMARY KEY, name VARCHAR, type VARCHAR, exchange VARCHAR,
    currency VARCHAR, sector VARCHAR, universe VARCHAR);
ALTER TABLE instruments ADD COLUMN IF NOT EXISTS asset_class VARCHAR;
CREATE TABLE IF NOT EXISTS prices(
    ticker VARCHAR, date DATE, open DOUBLE, high DOUBLE, low DOUBLE, close DOUBLE,
    adj_close DOUBLE, volume DOUBLE, PRIMARY KEY (ticker, date));
CREATE TABLE IF NOT EXISTS profiles(
    ticker VARCHAR PRIMARY KEY, summary VARCHAR, sector VARCHAR, industry VARCHAR, website VARCHAR,
    fetched TIMESTAMP);
CREATE TABLE IF NOT EXISTS portfolios(name VARCHAR PRIMARY KEY, cash DOUBLE, benchmark VARCHAR);
CREATE TABLE IF NOT EXISTS holdings(
    portfolio VARCHAR, ticker VARCHAR, units DOUBLE, cost_price DOUBLE, PRIMARY KEY (portfolio, ticker));
CREATE TABLE IF NOT EXISTS ingest_log(
    ticker VARCHAR PRIMARY KEY, first_date DATE, last_date DATE,
    last_run TIMESTAMP, status VARCHAR, error VARCHAR);
"""


_schema_ready: set[str] = set()  # per database file: tests point DB_PATH at a fresh one


@contextmanager
def connect():
    DATA.mkdir(exist_ok=True)
    path = str(DB_PATH)
    con = duckdb.connect(path)
    try:
        if path not in _schema_ready:
            con.execute(SCHEMA)
            _schema_ready.add(path)
        yield con
    finally:
        con.close()


def query(sql: str, params: list | None = None) -> pd.DataFrame:
    with connect() as con:
        return con.execute(sql, params or []).df()


def upsert(table: str, df: pd.DataFrame) -> None:
    if df.empty:
        return
    with connect() as con:
        con.register("incoming", df)
        con.execute(f"INSERT OR REPLACE INTO {table} BY NAME SELECT * FROM incoming")


def instruments(universes: list[str] | None = None) -> pd.DataFrame:
    if not universes:
        return query("SELECT * FROM instruments ORDER BY ticker")
    return query("SELECT * FROM instruments WHERE list_contains(?, universe) ORDER BY ticker", [universes])


def prices(tickers: list[str], start=None, end=None) -> pd.DataFrame:
    df = query(
        """SELECT * FROM prices WHERE list_contains(?, ticker)
           AND date >= coalesce(?::DATE, '1900-01-01') AND date <= coalesce(?::DATE, '2999-01-01')
           ORDER BY ticker, date""",
        [list(tickers), start, end],
    )
    df["date"] = pd.to_datetime(df["date"])
    return df


def date_ranges(tickers: list[str]) -> pd.DataFrame:
    return query(
        "SELECT ticker, min(date) AS first_date, max(date) AS last_date FROM prices "
        "WHERE list_contains(?, ticker) GROUP BY ticker",
        [list(tickers)],
    )


def ingest_log() -> pd.DataFrame:
    return query(
        "SELECT l.*, i.name, i.universe FROM ingest_log l LEFT JOIN instruments i USING (ticker) "
        "ORDER BY status DESC, ticker"
    )


def delete(tickers: list[str]) -> None:
    with connect() as con:
        for table in ("prices", "ingest_log", "instruments"):
            con.execute(f"DELETE FROM {table} WHERE list_contains(?, ticker)", [list(tickers)])


def custom() -> list[str]:
    return instruments(["custom"]).ticker.tolist()


def coverage() -> pd.DataFrame:
    """Per universe: how many instruments are held, how many have prices, and how current they are."""
    return query(
        """SELECT i.universe, count(*) AS instruments, count(p.ticker) AS with_data, max(p.last_date) AS last_date
           FROM instruments i
           LEFT JOIN (SELECT ticker, max(date) AS last_date FROM prices GROUP BY ticker) p USING (ticker)
           GROUP BY 1 ORDER BY 1"""
    )


def quality() -> pd.DataFrame:
    """Data problems worth knowing about, each with the tickers involved."""
    checks = {
        "No price data": "SELECT ticker FROM instruments WHERE ticker NOT IN (SELECT DISTINCT ticker FROM prices)",
        "Stale by over 7 days": """SELECT ticker FROM prices GROUP BY ticker
                                   HAVING max(date) < (SELECT max(date) FROM prices) - INTERVAL 7 DAY""",
        "Zero or negative prices": "SELECT DISTINCT ticker FROM prices WHERE adj_close <= 0",
        "Failed on last refresh": "SELECT ticker FROM ingest_log WHERE status <> 'ok'",
    }
    rows = [{"issue": name, "tickers": len(found), "examples": ", ".join(sorted(found.ticker)[:12])}
            for name, sql in checks.items() for found in [query(sql)]]
    return pd.DataFrame(rows)


def stats() -> dict:
    counts = query(
        "SELECT (SELECT count(*) FROM instruments) AS instruments, (SELECT count(*) FROM prices) AS rows, "
        "(SELECT count(DISTINCT ticker) FROM prices) AS tickers, (SELECT max(last_run) FROM ingest_log) AS last_run"
    ).iloc[0].to_dict()
    return counts | {"size_mb": DB_PATH.stat().st_size / 1e6 if DB_PATH.exists() else 0}
