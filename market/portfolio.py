"""Portfolios: holdings stored in DuckDB, valuation, contribution to return, risk decomposition and stress testing."""
import numpy as np
import pandas as pd

from market import store, stress
from market.stats.core import periods_per_year

COLUMNS = ["ticker", "units", "cost_price"]


def names() -> list[str]:
    return store.query("SELECT name FROM portfolios ORDER BY name")["name"].tolist()


def load(name: str) -> tuple[pd.DataFrame, pd.Series]:
    """Holdings and settings (cash, benchmark) for one portfolio."""
    holdings = store.query("SELECT ticker, units, cost_price FROM holdings WHERE portfolio = ? ORDER BY ticker", [name])
    meta = store.query("SELECT * FROM portfolios WHERE name = ?", [name])
    default = pd.Series({"name": name, "cash": 0.0, "benchmark": "^AXJO"})
    return holdings, meta.iloc[0] if len(meta) else default


def save(name: str, holdings: pd.DataFrame, cash: float, benchmark: str) -> None:
    holdings = holdings.dropna(subset=["ticker"]).query("ticker != '' and units > 0")
    with store.connect() as con:
        con.execute("DELETE FROM holdings WHERE portfolio = ?", [name])
        con.register("incoming", holdings.assign(portfolio=name)[["portfolio", *COLUMNS]])
        con.execute("INSERT INTO holdings SELECT * FROM incoming")
        con.execute("INSERT OR REPLACE INTO portfolios VALUES (?, ?, ?)", [name, float(cash), benchmark])


def delete(name: str) -> None:
    with store.connect() as con:
        con.execute("DELETE FROM holdings WHERE portfolio = ?", [name])
        con.execute("DELETE FROM portfolios WHERE name = ?", [name])


def values(holdings: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    """Market value of each holding through time (units × price, carried over non-trading days).

    The series starts where every holding has a price. Starting earlier would let a holding that
    listed mid-window arrive as a jump in portfolio value, which reads as a return that never happened.
    """
    units = holdings.set_index("ticker").units
    held = [t for t in units.index if t in prices]
    wide = prices[held].ffill()
    starts = [wide[t].first_valid_index() for t in held if wide[t].first_valid_index() is not None]
    wide = wide.loc[max(starts):] if starts else wide
    return wide.mul(units[held], axis=1).dropna(how="all")


def shortest_history(holdings: pd.DataFrame, prices: pd.DataFrame) -> str | None:
    """The holding whose prices start latest, which is what limits the common window."""
    held = [t for t in holdings.ticker if t in prices]
    starts = {t: prices[t].first_valid_index() for t in held}
    starts = {t: d for t, d in starts.items() if d is not None}
    return max(starts, key=starts.get) if starts else None


def concentration(weights: pd.Series) -> dict:
    """How much of the portfolio sits in few names: largest, top five, and the effective count.

    Largest and top five are portfolio weights, so cash is in the denominator and they match the
    positions table. The effective count is taken over the invested weights alone, since counting
    cash as a holding would flatter it.
    """
    w = weights.sort_values(ascending=False)
    invested = w / w.sum()
    return {"largest": w.iloc[0], "top_5": w.head(5).sum(), "effective_holdings": 1 / (invested ** 2).sum()}


def by_group(table: pd.DataFrame, instruments: pd.DataFrame, column: str) -> pd.DataFrame:
    """Value and weight grouped by an instrument attribute, such as asset class or sector."""
    group = instruments.set_index("ticker")[column].reindex(table.index).fillna("Unclassified")
    out = table.assign(group=group).groupby("group")[["value", "weight"]].sum()
    return out.sort_values("value", ascending=False)


def positions(holdings: pd.DataFrame, values: pd.DataFrame, cash: float) -> pd.DataFrame:
    """Current value, weight, cost and unrealised profit per holding."""
    latest = values.ffill().iloc[-1]
    table = holdings.set_index("ticker").loc[latest.index].assign(value=latest)
    table["cost"] = table.units * table.cost_price
    table["profit"] = table.value - table.cost
    table["return"] = table.value / table.cost - 1
    table["weight"] = table.value / (latest.sum() + cash)
    return table.rename_axis("ticker").sort_values("value", ascending=False)


def contributions(values: pd.DataFrame, cash: float) -> pd.DataFrame:
    """Each holding's share of the portfolio's return over the window (they sum to the total)."""
    first, last = values.ffill().bfill().iloc[0], values.ffill().iloc[-1]
    start_value = first.sum() + cash
    table = pd.DataFrame({"start_value": first, "end_value": last})
    table["start_weight"] = first / start_value
    table["return"] = last / first - 1
    table["contribution"] = (last - first) / start_value
    return table.sort_values("contribution", ascending=False)


def series(values: pd.DataFrame, cash: float) -> pd.Series:
    """Total portfolio value through time, including cash."""
    return values.ffill().sum(axis=1) + cash


def risk_contributions(returns: pd.DataFrame, weights: pd.Series) -> pd.DataFrame:
    """Marginal and percentage contribution to portfolio volatility (weights are normalised)."""
    r = returns.dropna()
    w = weights.reindex(r.columns).fillna(0)
    w = w / w.sum()
    cov = r.cov() * periods_per_year(r.index)
    variance = float(w @ cov @ w)
    if variance <= 0:
        return pd.DataFrame()
    vol = np.sqrt(variance)
    marginal = cov @ w / vol
    contribution = w * marginal
    return pd.DataFrame({"weight": w, "vol": np.sqrt(np.diag(cov)), "marginal": marginal,
                         "contribution": contribution, "share_of_risk": contribution / vol}
                        ).sort_values("share_of_risk", ascending=False)


def stress_history(prices: pd.DataFrame, weights: pd.Series, periods=None) -> pd.DataFrame:
    """Today's weights applied to each past stress period: return, worst point and recovery."""
    rows = {}
    for period in periods or stress.catalogue():
        available = [t for t in weights.index if t in prices and prices[t].first_valid_index() is not None
                     and prices[t].first_valid_index() <= pd.Timestamp(period.start)]
        if not available:
            continue
        path = stress.stress_test(prices, weights[available], period)
        if len(path) < 2:
            continue
        rows[period.name] = {"start": pd.Timestamp(period.start), "end": pd.Timestamp(period.end),
                             "return": path.iloc[-1] - 1, "worst": path.min() - 1,
                             "covered": len(available) / len(weights)}
    return pd.DataFrame.from_dict(rows, orient="index").sort_values("return")
