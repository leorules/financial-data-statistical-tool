import numpy as np
import pandas as pd
import pytest

from market import portfolio, stress


@pytest.fixture
def holdings():
    return pd.DataFrame({"ticker": ["A", "B"], "units": [100.0, 50.0], "cost_price": [8.0, 20.0]})


@pytest.fixture
def prices(dates):
    return pd.DataFrame({"A": np.linspace(10, 20, len(dates)), "B": np.linspace(20, 10, len(dates))}, index=dates)


def test_values_positions_and_contributions(holdings, prices):
    values = portfolio.values(holdings, prices)
    assert values.iloc[0].tolist() == [1000, 1000] and values.iloc[-1].tolist() == [2000, 500]

    table = portfolio.positions(holdings, values, cash=500)
    assert table.loc["A", "value"] == 2000 and table.loc["A", "cost"] == 800
    assert table.loc["A", "profit"] == 1200 and table.loc["A", "weight"] == pytest.approx(2000 / 3000)

    total = portfolio.series(values, cash=500)
    assert total.iloc[0] == 2500 and total.iloc[-1] == 3000

    contribution = portfolio.contributions(values, cash=500)
    assert contribution.loc["A", "contribution"] == pytest.approx(1000 / 2500)
    assert contribution.loc["B", "contribution"] == pytest.approx(-500 / 2500)
    assert contribution.contribution.sum() == pytest.approx(total.iloc[-1] / total.iloc[0] - 1)


def test_risk_contributions_sum_to_portfolio_volatility(rng, dates):
    returns = pd.DataFrame({"A": rng.normal(0, 0.02, len(dates)), "B": rng.normal(0, 0.005, len(dates))}, index=dates)
    weights = pd.Series({"A": 0.5, "B": 0.5})
    table = portfolio.risk_contributions(returns, weights)
    assert table.share_of_risk.sum() == pytest.approx(1.0)
    assert table.index[0] == "A"  # the volatile holding carries most of the risk
    assert table.loc["A", "share_of_risk"] > table.loc["A", "weight"]


def test_stress_history_uses_holdings_with_history(prices):
    weights = pd.Series({"A": 0.7, "B": 0.3})
    period = stress.StressPeriod("Slump", str(prices.index[100].date()), str(prices.index[200].date()), "Crash", "")
    table = portfolio.stress_history(prices, weights, [period])
    assert table.loc["Slump", "covered"] == 1.0
    assert table.loc["Slump", "worst"] <= table.loc["Slump", "return"]

    later = prices.copy()
    later.loc[later.index[:150], "B"] = np.nan  # B did not exist when the period began
    assert portfolio.stress_history(later, weights, [period]).loc["Slump", "covered"] == 0.5


def test_save_load_and_delete(temp_db, holdings):
    portfolio.save("Main", holdings, cash=250.0, benchmark="^AXJO")
    loaded, meta = portfolio.load("Main")
    assert loaded.ticker.tolist() == ["A", "B"] and meta.cash == 250.0 and meta.benchmark == "^AXJO"
    assert portfolio.names() == ["Main"]

    portfolio.save("Main", holdings.iloc[:1], cash=0.0, benchmark="^GSPC")
    loaded, meta = portfolio.load("Main")
    assert loaded.ticker.tolist() == ["A"] and meta.benchmark == "^GSPC"

    portfolio.delete("Main")
    assert portfolio.names() == [] and portfolio.load("Main")[0].empty


def test_a_late_listing_holding_does_not_invent_a_return():
    idx = pd.bdate_range("2024-01-01", periods=200)
    prices = pd.DataFrame({"OLD": np.linspace(10, 11, 200), "NEW": np.nan}, index=idx)
    prices.loc[idx[100]:, "NEW"] = np.linspace(20, 21, 100)
    held = pd.DataFrame({"ticker": ["OLD", "NEW"], "units": [1000.0, 500.0], "cost_price": [10.0, 20.0]})

    values = portfolio.values(held, prices)
    assert values.index[0] == idx[100], "the series starts where every holding has a price"
    total = portfolio.series(values, 0.0)
    step = total.pct_change().abs().max()
    assert step < 0.01, "both holdings are straight lines, so no day should jump"
    assert portfolio.shortest_history(held, prices) == "NEW"


def test_concentration_and_grouping():
    weights = pd.Series({"A": 0.4, "B": 0.3, "C": 0.2, "D": 0.1})
    c = portfolio.concentration(weights)
    assert c["largest"] == 0.4 and c["top_5"] == pytest.approx(1.0)
    assert c["effective_holdings"] == pytest.approx(1 / 0.30)  # 0.16+0.09+0.04+0.01, weights already sum to 1
    # Cash sits outside the invested weights, so it cannot inflate the effective count.
    half_cash = pd.Series({"A": 0.2, "B": 0.15, "C": 0.1, "D": 0.05})
    assert portfolio.concentration(half_cash)["effective_holdings"] == pytest.approx(1 / 0.30)
    equal = portfolio.concentration(pd.Series(0.25, index=list("ABCD")))
    assert equal["effective_holdings"] == pytest.approx(4.0)

    table = pd.DataFrame({"value": [60.0, 40.0], "weight": [0.6, 0.4]}, index=["A", "B"])
    inst = pd.DataFrame({"ticker": ["A", "B"], "asset_class": ["Equities", "Equities"], "sector": ["Banks", "Miners"]})
    by_class = portfolio.by_group(table, inst, "asset_class")
    assert by_class.loc["Equities", "weight"] == pytest.approx(1.0)
    assert portfolio.by_group(table, inst, "sector").index.tolist() == ["Banks", "Miners"]


def test_a_portfolio_remembers_its_objective(temp_db):
    holdings = pd.DataFrame({"ticker": ["BHP.AX"], "units": [100.0], "cost_price": [40.0]})
    portfolio.save("obj", holdings, 500.0, "^AXJO", objective_margin=0.045, objective_years=7, cpi_region="USA")
    _, meta = portfolio.load("obj")
    assert (meta.objective_margin, meta.objective_years, meta.cpi_region) == (0.045, 7, "USA")

    # Saving without an objective falls back to the common super-fund target, not to nulls.
    portfolio.save("plain", holdings, 0.0, "^GSPC")
    _, plain = portfolio.load("plain")
    assert (plain.objective_margin, plain.objective_years, plain.cpi_region) == (0.035, 10, "AUS")
    assert portfolio.load("never-saved")[1].objective_years == 10
