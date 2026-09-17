import numpy as np
import pandas as pd
import pytest

from market import stress


@pytest.fixture
def crash():
    return stress.StressPeriod("Test crash", "2020-02-03", "2020-02-14", "Crash", "")


def test_catalogue_dates_are_valid():
    for event in stress.BUILT_IN:
        assert pd.Timestamp(event.start) < pd.Timestamp(event.end), event.name
    names = [e.name for e in stress.BUILT_IN]
    assert len(names) == len(set(names)) and "COVID-19 crash" in names and "Global Financial Crisis (GFC)" in names
    assert min(e.start for e in stress.BUILT_IN) < "1930", "catalogue should reach back to the Great Depression"


def test_impact_measures_fall_drawdown_and_recovery(crash):
    dates = pd.bdate_range("2019-01-01", "2020-06-30")
    price = pd.Series(100.0, index=dates)
    window = (dates > "2020-02-03") & (dates <= "2020-02-14")
    price[window] = np.linspace(95, 70, window.sum())
    price[dates > "2020-02-14"] = np.linspace(72, 110, (dates > "2020-02-14").sum())
    table = stress.impact(pd.DataFrame({"A": price, "LATE": price[dates > "2020-03-01"]}), crash)
    row = table.loc["A"]
    assert list(table.index) == ["A"]  # LATE has no history before the event
    assert row.event_return == pytest.approx(-0.30) and row.trough_return == pytest.approx(-0.30)
    assert row.max_drawdown == pytest.approx(-0.30) and row.worst_day == pytest.approx(-0.05)
    assert row.recovery_date > pd.Timestamp("2020-02-14") and row.days_to_recover > 11


def test_stress_test_and_correlation_shift(crash):
    dates = pd.bdate_range("2019-01-01", "2020-03-31")
    rng = np.random.default_rng(0)
    common = rng.normal(0, 0.01, len(dates))
    prices = pd.DataFrame({"A": 100 * np.cumprod(1 + common + rng.normal(0, 0.01, len(dates))),
                           "B": 100 * np.cumprod(1 + rng.normal(0, 0.01, len(dates)))}, index=dates)
    path = stress.stress_test(prices, pd.Series({"A": 3.0, "B": 1.0}), crash)
    start = prices[prices.index <= "2020-02-03"].iloc[-1]
    expected = 0.75 * prices.A / start.A + 0.25 * prices.B / start.B
    assert path.iloc[-1] == pytest.approx(expected["2020-02-14"])
    before, during = stress.correlation_shift(prices, crash)
    assert -1 <= before <= 1 and -1 <= during <= 1


def test_custom_stress_periods_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(stress, "PERIODS_PATH", tmp_path / "stress.json")
    monkeypatch.setattr(stress, "LEGACY_PATH", tmp_path / "legacy.json")
    stress.save_custom(stress.StressPeriod("My stress period", "2021-01-04", "2021-01-29", "Custom", "note"))
    assert [e.name for e in stress.custom()] == ["My stress period"] and stress.custom()[0].custom
    assert "My stress period" in [e.name for e in stress.catalogue()]
    stress.delete_custom("My stress period")
    assert stress.custom() == []
