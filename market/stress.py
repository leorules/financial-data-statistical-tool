"""Stress periods (crashes, crises, shocks): a built-in catalogue, user-defined periods, and impact analysis."""
import json
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from market.config import DATA
from market.stats.core import periods_per_year

PERIODS_PATH = DATA / "stress_periods.json"
LEGACY_PATH = DATA / "events.json"


@dataclass(frozen=True)
class StressPeriod:
    name: str
    start: str
    end: str
    category: str
    description: str
    custom: bool = False

    @property
    def span(self) -> str:
        start, end = pd.Timestamp(self.start), pd.Timestamp(self.end)
        return f"{start:%b %Y}" if (start.year, start.month) == (end.year, end.month) else f"{start:%b %Y} – {end:%b %Y}"

    @property
    def label(self) -> str:
        return f"{self.name} ({self.span})"


# Windows run from the S&P 500 closing peak to its closing trough unless the description says otherwise.
BUILT_IN = [
    StressPeriod("Wall Street Crash & Great Depression", "1929-09-03", "1932-07-08", "Depression",
          "The 1929 crash and the depression that followed: US shares lost about 86% of their value over nearly three "
          "years, amid bank failures, deflation and mass unemployment. It took until 1954 to regain the 1929 peak."),
    StressPeriod("1937-38 recession", "1937-03-06", "1938-03-31", "Bear market",
          "A second slump inside the Depression era after fiscal and monetary tightening was withdrawn too early; US "
          "shares roughly halved."),
    StressPeriod("1973-74 oil crisis", "1973-01-11", "1974-10-03", "Bear market",
          "The OPEC oil embargo, the collapse of Bretton Woods and double-digit inflation produced the worst bear "
          "market since the 1930s, with shares down about 48% in nominal terms and far more after inflation."),
    StressPeriod("1980-82 Volcker recession", "1980-11-28", "1982-08-12", "Bear market",
          "US interest rates above 15% under Paul Volcker crushed inflation but caused a deep recession; the bear "
          "market ended in August 1982 at the start of a long bull run."),
    StressPeriod("Black Monday", "1987-08-25", "1987-12-04", "Crash",
          "The October 1987 crash: the Dow fell 22.6% on 19 October, still its largest one-day percentage fall. "
          "Portfolio insurance and program selling amplified the decline worldwide."),
    StressPeriod("Russia default & LTCM", "1998-07-17", "1998-08-31", "Crisis",
          "Russia defaulted on its domestic debt and devalued the rouble, triggering the near-collapse of the hedge "
          "fund Long-Term Capital Management and a flight to quality."),
    StressPeriod("Dot-com bust", "2000-03-24", "2002-10-09", "Bear market",
          "The technology bubble deflated; the Nasdaq Composite lost almost 80% from its peak and the S&P 500 about half."),
    StressPeriod("September 11 attacks", "2001-09-10", "2001-09-21", "Shock",
          "US markets closed for four trading days after the attacks and fell sharply when they reopened on 17 September."),
    StressPeriod("Global Financial Crisis (GFC)", "2007-10-09", "2009-03-09", "Crisis",
          "The US subprime mortgage collapse spread into a global banking crisis, peaking with Lehman Brothers' failure "
          "in September 2008. The S&P 500 fell about 57% and the ASX 200 more than half."),
    StressPeriod("Euro debt crisis & US downgrade", "2011-04-29", "2011-10-03", "Crisis",
          "Sovereign debt stress in Greece, Italy and Spain combined with S&P's first-ever downgrade of the US credit "
          "rating in August 2011."),
    StressPeriod("Taper tantrum", "2013-05-22", "2013-06-24", "Shock",
          "Bond yields jumped after the Federal Reserve signalled it could slow its asset purchases. Window from Ben "
          "Bernanke's testimony to the equity low."),
    StressPeriod("China slowdown & oil collapse", "2015-05-21", "2016-02-11", "Bear market",
          "China's equity crash and yuan devaluation, plus oil falling below $30 a barrel, hit commodities, emerging "
          "markets and resource stocks such as the Australian miners."),
    StressPeriod("Brexit vote", "2016-06-23", "2016-06-27", "Shock",
          "The UK's vote to leave the European Union caused a sharp two-day sell-off in global equities and the pound. "
          "Window from the referendum to the low."),
    StressPeriod("Volmageddon", "2018-01-26", "2018-02-08", "Shock",
          "A spike in volatility wiped out short-VIX products and caused a rapid 10% equity correction."),
    StressPeriod("Q4 2018 sell-off", "2018-09-20", "2018-12-24", "Bear market",
          "Rising US rates and trade-war fears drove a near-20% fall in the S&P 500, bottoming on Christmas Eve."),
    StressPeriod("COVID-19 crash", "2020-02-19", "2020-03-23", "Crash",
          "The fastest bear market on record as the pandemic spread: the S&P 500 fell about 34% in 23 trading days "
          "before emergency central bank and government support sparked a rebound."),
    StressPeriod("2022 rate-hike bear market", "2022-01-03", "2022-10-12", "Bear market",
          "Inflation forced the fastest central bank tightening in decades; both equities and bonds fell sharply, "
          "undermining traditional diversification."),
    StressPeriod("US regional bank crisis", "2023-03-08", "2023-05-01", "Crisis",
          "Silicon Valley Bank's losses and collapse triggered runs on regional banks. Window from SVB's announcement "
          "to First Republic's seizure."),
    StressPeriod("Yen carry-trade unwind", "2024-07-31", "2024-08-05", "Shock",
          "A Bank of Japan rate rise and weak US data unwound yen-funded trades; the Nikkei 225 fell 12.4% on 5 August. "
          "Window from the BoJ decision to the low."),
    StressPeriod("April 2025 tariff shock", "2025-04-02", "2025-04-08", "Shock",
          "Sweeping US 'Liberation Day' tariffs triggered a global sell-off, reversed partly by a 90-day pause "
          "announced on 9 April. Window from the announcement to the low."),
]


def custom() -> list[StressPeriod]:
    path = PERIODS_PATH if PERIODS_PATH.exists() else LEGACY_PATH
    if not path.exists():
        return []
    return [StressPeriod(**p, custom=True) for p in json.loads(path.read_text(encoding="utf-8"))]


def catalogue() -> list[StressPeriod]:
    return sorted(BUILT_IN + custom(), key=lambda e: e.start)


def _write(periods: list[StressPeriod]) -> None:
    PERIODS_PATH.parent.mkdir(parents=True, exist_ok=True)
    records = [{k: v for k, v in asdict(p).items() if k != "custom"} for p in periods]
    PERIODS_PATH.write_text(json.dumps(records, indent=2), encoding="utf-8")


def save_custom(period: StressPeriod) -> None:
    _write([p for p in custom() if p.name != period.name] + [period])


def delete_custom(name: str) -> None:
    _write([p for p in custom() if p.name != name])


def overlapping(start, end) -> list[StressPeriod]:
    """Stress periods that overlap a date range (None start means full history)."""
    lo = pd.Timestamp(start) if start else pd.Timestamp.min
    return [p for p in catalogue() if pd.Timestamp(p.end) >= lo and pd.Timestamp(p.start) <= pd.Timestamp(end)]


# --- analysis --------------------------------------------------------------------------------------------------

def impact(prices: pd.DataFrame, period: StressPeriod, lookback: int = 252) -> pd.DataFrame:
    """Per-series statistics from a wide price matrix covering the stress period plus history either side."""
    start, end = pd.Timestamp(period.start), pd.Timestamp(period.end)
    rows = {}
    for col in prices:
        p = prices[col].dropna()
        before, during = p[p.index <= start], p[(p.index >= start) & (p.index <= end)]
        if before.empty or len(during) < 2:
            continue
        base = before.iloc[-1]
        path = pd.concat([before.iloc[-1:], during[during.index > start]])
        r = p.pct_change()
        ppy = periods_per_year(p.index)
        prior = r[r.index <= start].iloc[-lookback:]
        inside = r[(r.index > start) & (r.index <= end)]
        after = p[p.index > path.idxmin()]
        recovered = after[after >= base]
        rows[col] = {
            "event_return": path.iloc[-1] / base - 1,
            "max_drawdown": (path / path.cummax() - 1).min(),
            "trough_return": path.min() / base - 1,
            "trough_date": path.idxmin(),
            "worst_day": inside.min(),
            "vol_before": prior.std() * np.sqrt(ppy),
            "vol_during": inside.std() * np.sqrt(ppy) if len(inside) > 1 else np.nan,
            "recovery_date": recovered.index[0] if len(recovered) else pd.NaT,
        }
    table = pd.DataFrame.from_dict(rows, orient="index")
    if table.empty:
        return table
    table["vol_ratio"] = table.vol_during / table.vol_before
    table["days_to_recover"] = (table.recovery_date - start).dt.days
    return table


def correlation_shift(prices: pd.DataFrame, period: StressPeriod, lookback: int = 252) -> tuple[float, float]:
    """Average pairwise correlation of returns in the year before the stress period versus during it."""
    r = prices.pct_change(fill_method=None)
    start, end = pd.Timestamp(period.start), pd.Timestamp(period.end)
    before = r[r.index <= start].iloc[-lookback:].dropna()
    during = r[(r.index > start) & (r.index <= end)].dropna()

    def average(frame: pd.DataFrame) -> float:
        if len(frame) < 3 or frame.shape[1] < 2:
            return np.nan
        corr = frame.corr().to_numpy()
        return float(corr[np.triu_indices_from(corr, k=1)].mean())

    return average(before), average(during)


def stress_test(prices: pd.DataFrame, weights: pd.Series, period: StressPeriod, after_days: int = 0) -> pd.Series:
    """Buy-and-hold basket from the period start until `after_days` past its end, indexed to 1 (weights normalised)."""
    start, end = pd.Timestamp(period.start), pd.Timestamp(period.end) + pd.Timedelta(days=after_days)
    weights = weights[weights > 0] / weights[weights > 0].sum()
    window = prices[weights.index].ffill()
    window = window[(window.index >= start) & (window.index <= end)]
    base = prices[weights.index].ffill()[prices.index <= start].iloc[-1]
    return (window / base * weights).sum(axis=1)
