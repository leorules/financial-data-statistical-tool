from datetime import date

import pandas as pd
import yfinance as yf

from market.providers.base import COLUMNS

RENAME = {"Date": "date", "Ticker": "ticker", "Open": "open", "High": "high", "Low": "low",
          "Close": "close", "Adj Close": "adj_close", "Volume": "volume"}


def to_long(raw: pd.DataFrame) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame(columns=COLUMNS)
    df = raw.stack(level=0).reset_index().rename(columns=RENAME)
    df = df.dropna(subset=["close"])
    df["date"] = pd.to_datetime(df["date"]).dt.date
    return df[COLUMNS]


class Yahoo:
    def fetch(self, tickers: list[str], start: date | None) -> pd.DataFrame:
        period = {"start": start} if start else {"period": "max"}
        raw = yf.download(tickers, auto_adjust=False, group_by="ticker", actions=False,
                          progress=False, threads=True, **period)
        return to_long(raw)
