import time
from datetime import date, datetime, timedelta
from typing import Callable

import pandas as pd

from market import store
from market.config import BATCH_PAUSE, BATCH_SIZE, REFETCH_DAYS, RETRIES
from market.providers.base import COLUMNS, PriceProvider
from market.providers.yahoo import Yahoo


def start_date(last) -> date | None:
    """Re-fetch a few days to pick up corrections; floor to Monday so tickers share batches."""
    if last is None or pd.isna(last):
        return None
    start = pd.Timestamp(last).date() - timedelta(days=REFETCH_DAYS)
    return start - timedelta(days=start.weekday())


def plan_batches(tickers: list[str], last_dates: dict) -> list[tuple[date | None, list[str]]]:
    groups: dict = {}
    for t in tickers:
        groups.setdefault(start_date(last_dates.get(t)), []).append(t)
    return [(start, ts[i:i + BATCH_SIZE]) for start, ts in groups.items() for i in range(0, len(ts), BATCH_SIZE)]


def fetch_with_retry(provider: PriceProvider, tickers: list[str], start: date | None) -> pd.DataFrame:
    for attempt in range(RETRIES):
        try:
            return provider.fetch(tickers, start)
        except Exception:
            if attempt == RETRIES - 1:
                raise
            time.sleep(2 ** attempt)


def refresh(tickers: list[str], provider: PriceProvider | None = None,
            on_progress: Callable[[float, str], None] | None = None, pause: float = BATCH_PAUSE) -> pd.DataFrame:
    """Fetch new bars for tickers, upsert them, and record per-ticker status in ingest_log."""
    provider = provider or Yahoo()
    tickers = list(dict.fromkeys(tickers))
    last = store.date_ranges(tickers).set_index("ticker")["last_date"].to_dict()
    batches = plan_batches(tickers, last)
    status = {}
    for i, (start, batch) in enumerate(batches):
        try:
            df, error = fetch_with_retry(provider, batch, start), None
        except Exception as e:
            df, error = pd.DataFrame(columns=COLUMNS), f"{type(e).__name__}: {e}"
        store.upsert("prices", df)
        got = set(df["ticker"])
        status |= {t: ("ok", None) if t in got else ("error", error or "no data returned") for t in batch}
        if on_progress:
            on_progress((i + 1) / len(batches), f"{batch[0]}..{batch[-1]} ({len(got)}/{len(batch)} ok)")
        if pause and i < len(batches) - 1:
            time.sleep(pause)

    log = pd.DataFrame([{"ticker": t, "status": s, "error": e} for t, (s, e) in status.items()])
    log = log.merge(store.date_ranges(tickers), on="ticker", how="left").assign(last_run=datetime.now())
    store.upsert("ingest_log", log)
    return log
