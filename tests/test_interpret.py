import numpy as np
import pandas as pd
import pytest

from market import interpret, stats


@pytest.fixture
def market(rng):
    dates = pd.bdate_range("2016-01-01", periods=2500)
    bench = pd.Series(rng.normal(0.0004, 0.01, len(dates)), index=dates, name="BENCH")
    fat = pd.Series(rng.standard_t(3, len(dates)) * 0.01 + 0.5 * bench, index=dates, name="FAT")
    calm = pd.Series(rng.normal(0.0002, 0.004, len(dates)), index=dates, name="CALM")
    return pd.concat([fat, calm, bench], axis=1)


def text(reading: interpret.Reading) -> str:
    return " ".join([reading.headline, *reading.findings, *reading.implications, *reading.caveats])


def test_wording_helpers():
    assert interpret.evidence(0.0004).startswith("very strong") and interpret.evidence(0.3).startswith("no real")
    assert interpret.corr_strength(-0.7) == "strong negative" and interpret.corr_strength(0.1) == "very weak positive"
    assert interpret.sharpe_level(-0.2).startswith("negative") and interpret.vol_level(0.5) == "very high"
    assert interpret.pct(0.1234, sign=True) == "+12.3%" and interpret.money(12345.6) == "$12,346"


def test_every_reading_term_has_a_definition(market):
    reading = interpret.descriptive(stats.descriptive.summary(market), "FAT", True, "D", market["FAT"])
    assert set(reading.definitions()) == set(reading.terms)


def test_descriptive_and_distribution(market):
    x = market["FAT"]
    reading = interpret.descriptive(stats.descriptive.summary(market), "FAT", True, "D", x)
    assert "Fat tails" in text(reading) and "$10,000" in text(reading) and "payoff ratio" in text(reading)
    assert "per unit of volatility" in text(reading)
    reading = interpret.distribution(stats.descriptive.describe(x), stats.distribution.normality(x),
                                     stats.distribution.fit(x), stats.distribution.var(x), "D", x)
    assert reading.headline.startswith("Returns are **not normally distributed**")
    assert "Student-t" in text(reading) and "times a year" in text(reading)


def test_risk_regression_and_correlation(market):
    table = stats.risk.summary(market, market["BENCH"])
    reading = interpret.risk(table, "FAT", "BENCH", market["FAT"])
    assert "Versus BENCH" in text(reading) and "Ranked by Sharpe" in text(reading)

    price = (1 + market["FAT"]).cumprod()
    res = stats.regression.fit(market["FAT"], market[["BENCH"]])
    trend, fitted = stats.regression.trend(price)
    reading = interpret.regression(stats.regression.coefficients(res), stats.regression.diagnostics(res), trend, "FAT",
                                   True, "D", price, fitted)
    assert "**BENCH** is the most important" in reading.headline and "systematic" in text(reading)

    corr = market.corr()
    explained, _ = stats.multivariate.pca(market)
    reading = interpret.correlation(corr, stats.correlation.pvalues(market), stats.correlation.pairs(corr), explained,
                                    "pearson", market)
    assert "independent bets" in reading.headline and "FAT & BENCH" in text(reading)


def test_hypothesis_timeseries_seasonality_compare(market):
    x, y = market["FAT"], market["CALM"]
    ci, _ = stats.hypothesis.bootstrap(x, stats.hypothesis.boot_mean, 200)
    reading = interpret.hypothesis(stats.hypothesis.mean_zero(x), stats.hypothesis.compare(x, y), ci, "mean", "FAT",
                                   "CALM", "D", x, y)
    assert "years) of data" in text(reading) and "volatility" in text(reading)

    level = np.log((1 + market).cumprod())
    tests = pd.concat([stats.timeseries.stationarity(level["FAT"]).assign(series="log price"),
                       stats.timeseries.stationarity(x).assign(series="returns"),
                       pd.DataFrame([stats.timeseries.ljung_box(x).to_dict() | {"series": "returns"},
                                     stats.timeseries.variance_ratio(x).to_dict() | {"series": "returns"}])])
    granger = stats.timeseries.granger(x, y).assign(direction="CALM → FAT")
    reading = interpret.timeseries(tests, 0.5, stats.timeseries.cointegration(level["FAT"], level["CALM"]), granger,
                                   "FAT", "CALM", "D", stats.timeseries.autocorr(x, 20), 0.4)
    assert "random walk" in reading.headline and "Granger" in text(reading)

    monthly = stats.timeseries.monthly_table(x)
    reading = interpret.seasonality([("month", *stats.timeseries.seasonality(x, "month"))], monthly, "FAT")
    assert "no reliable" in reading.headline and "consistent with chance" in text(reading)

    prices = (1 + market).cumprod()
    table = stats.risk.summary(market).assign(total_return=prices.iloc[-1] / prices.iloc[0] - 1)
    assert "grew to" in text(interpret.compare(table))
