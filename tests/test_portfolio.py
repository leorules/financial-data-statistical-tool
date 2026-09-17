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
