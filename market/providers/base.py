from datetime import date
from typing import Protocol

import pandas as pd

COLUMNS = ["ticker", "date", "open", "high", "low", "close", "adj_close", "volume"]


class PriceProvider(Protocol):
    def fetch(self, tickers: list[str], start: date | None) -> pd.DataFrame:
        """Daily bars in long format with COLUMNS; start=None means full history."""
