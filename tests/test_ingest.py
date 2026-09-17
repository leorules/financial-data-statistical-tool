from datetime import date

import pandas as pd

from market import ingest
from market.providers.yahoo import to_long


class FakeProvider:
    def __init__(self):
        self.calls = []

    def fetch(self, tickers, start):
        self.calls.append((tuple(tickers), start))
        dates = pd.bdate_range("2024-01-01", "2024-01-31")
        cols = pd.MultiIndex.from_product([tickers, ["Open", "High", "Low", "Close", "Adj Close", "Volume"]],
                                          names=["Ticker", "Price"])
        raw = pd.DataFrame(1.0, index=pd.Index(dates, name="Date"), columns=cols)
        if "BAD" in tickers:
            raw[("BAD", "Close")] = float("nan")
        return to_long(raw)


def test_refresh_upserts_idempotently_and_logs(temp_db):
    provider = FakeProvider()
    log = ingest.refresh(["AAA", "BAD"], provider, pause=0).set_index("ticker")
    assert log.loc["AAA", "status"] == "ok" and log.loc["BAD", "status"] == "error"
    assert provider.calls[0][1] is None

    ingest.refresh(["AAA"], provider, pause=0)
    assert temp_db.query("SELECT count(*) AS n FROM prices").n[0] == 23
    assert provider.calls[1][1] == date(2024, 1, 22)
    assert temp_db.ingest_log().set_index("ticker").loc["AAA", "last_date"] == pd.Timestamp("2024-01-31")


def test_plan_batches_groups_by_start():
    last = {"A": pd.Timestamp("2024-01-31"), "B": pd.Timestamp("2024-02-01")}
    batches = ingest.plan_batches(["A", "B", "C"], last)
    assert batches == [(date(2024, 1, 22), ["A", "B"]), (None, ["C"])]
