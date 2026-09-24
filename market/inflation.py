"""Consumer price indices for the major financial centres, from the OECD's public SDMX service.

Frequency follows how each office actually publishes: the ABS releases Australian CPI quarterly, so
taking the monthly series there would buy two years of history instead of sixty. Dates are stored at
the end of the period they measure, so a CPI reading lines up with market data for the same date.

FRED is the usual source for this and is unreachable from this machine; the OECD covers every hub
here except Singapore, Hong Kong and mainland China, which publish nothing comparable for free.
"""
import io

import pandas as pd
import requests

from market import store

SERVICE = "https://sdmx.oecd.org/public/rest/data/OECD.SDD.TPS,DSD_PRICES@DF_PRICES_ALL,1.0/"
HEADERS = {"User-Agent": "Mozilla/5.0 market-dashboard"}

# region code: (label, publication frequency, the centre it covers)
HUBS = {
    "AUS": ("Australia", "Q", "Sydney"),
    "USA": ("United States", "M", "New York"),
    "GBR": ("United Kingdom", "M", "London"),
    "EA20": ("Euro area", "M", "Frankfurt"),
    "JPN": ("Japan", "M", "Tokyo"),
    "CHE": ("Switzerland", "M", "Zurich"),
    "CAN": ("Canada", "M", "Toronto"),
    "KOR": ("South Korea", "M", "Seoul"),
}


def fetch(region: str) -> pd.DataFrame:
    """The CPI index level for one region, as (region, date, cpi) rows."""
    label, freq, _ = HUBS[region]
    start = "1960-Q1" if freq == "Q" else "1960-01"
    url = f"{SERVICE}{region}.{freq}.N.CPI.._T.N._Z?startPeriod={start}&format=csvfile"
    response = requests.get(url, headers=HEADERS, timeout=60)
    response.raise_for_status()
    raw = pd.read_csv(io.StringIO(response.text))
    period = pd.PeriodIndex(raw.TIME_PERIOD, freq=freq)
    return pd.DataFrame({"region": region, "date": period.to_timestamp(how="end").normalize(),
                         "cpi": raw.OBS_VALUE.astype(float)}).dropna().sort_values("date")


def refresh(regions: list[str] | None = None, on_progress=None) -> pd.DataFrame:
    """Download and store each region, recording what arrived and what failed."""
    rows = []
    wanted = regions or list(HUBS)
    for i, region in enumerate(wanted):
        try:
            data = fetch(region)
            store.upsert("inflation", data)
            rows.append({"region": region, "observations": len(data), "latest": data.date.iloc[-1], "error": None})
        except Exception as e:
            rows.append({"region": region, "observations": 0, "latest": pd.NaT, "error": f"{type(e).__name__}: {e}"})
        if on_progress:
            on_progress((i + 1) / len(wanted), f"{HUBS[region][0]} ({i + 1}/{len(wanted)})")
    return pd.DataFrame(rows)


def series(region: str = "AUS") -> pd.Series:
    """CPI index level through time."""
    df = store.query("SELECT date, cpi FROM inflation WHERE region = ? ORDER BY date", [region])
    return df.set_index(pd.to_datetime(df.date)).cpi if len(df) else pd.Series(dtype=float)


def yoy(region: str = "AUS") -> pd.Series:
    """Inflation rate: the change in the index over the four quarters or twelve months to each date."""
    cpi = series(region)
    return cpi.pct_change(4 if HUBS[region][1] == "Q" else 12).dropna()


def deflate(values: pd.Series, region: str = "AUS") -> pd.Series:
    """Restate a series in the purchasing power of its last date, so growth is real rather than nominal."""
    cpi = series(region).reindex(values.index.union(series(region).index)).interpolate().reindex(values.index).ffill()
    return values * cpi.iloc[-1] / cpi if cpi.notna().any() else values


def coverage() -> pd.DataFrame:
    """What is stored per region, for the Data Manager."""
    held = store.query("SELECT region, count(*) AS observations, min(date) AS first, max(date) AS latest "
                       "FROM inflation GROUP BY region")
    rows = pd.DataFrame([{"region": r, "hub": f"{hub} · {label}", "frequency": "Quarterly" if f == "Q" else "Monthly"}
                         for r, (label, f, hub) in HUBS.items()])
    return rows.merge(held, on="region", how="left").fillna({"observations": 0})
