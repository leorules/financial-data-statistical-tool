import numpy as np
import pandas as pd
import pytest
from scipy import stats as st

from market import factors, inflation, returns, stats, tearsheet


@pytest.fixture
def xy(rng, dates):
    x = pd.Series(rng.normal(0, 0.01, len(dates)), index=dates, name="x")
    y = (2 * x + rng.normal(0, 0.001, len(dates))).rename("y")
    z = pd.Series(rng.normal(0, 0.01, len(dates)), index=dates, name="z")
    return pd.concat([x, y, z], axis=1)


def test_periods_per_year(dates):
    assert stats.periods_per_year(dates) == 252
    assert stats.periods_per_year(pd.date_range("2020", periods=50, freq="W-FRI")) == 52
    assert stats.periods_per_year(pd.date_range("2020", periods=50, freq="ME")) == 12


def test_descriptive_matches_numpy(xy):
    s = xy["x"]
    d = stats.descriptive.describe(s)
    assert d["mean"] == pytest.approx(s.mean()) and d["std"] == pytest.approx(np.std(s, ddof=1))
    assert d["skew"] == pytest.approx(st.skew(s, bias=False))
    assert d["kurtosis"] == pytest.approx(st.kurtosis(s, bias=False))
    assert d["cum_return"] == pytest.approx((1 + s).prod() - 1)
    assert stats.descriptive.summary(xy).shape[0] == 3


def test_streaks_and_cagr():
    idx = pd.bdate_range("2020-01-01", periods=6)
    r = pd.Series([0.1, 0.1, -0.1, 0.1, 0.1, 0.1], index=idx)
    d = stats.descriptive.describe(r, periods=6)
    assert d["win_streak"] == 3 and d["loss_streak"] == 1
    assert d["cagr"] == pytest.approx(1.1**5 * 0.9 - 1)


def test_correlation_and_regression(xy):
    corr = stats.correlation.matrix(xy)
    assert corr.loc["x", "y"] > 0.99 and abs(corr.loc["x", "z"]) < 0.1
    assert stats.correlation.pvalues(xy).loc["x", "z"] > 0.05
    pairs = stats.correlation.pairs(corr)
    assert pairs.iloc[0][["a", "b"]].tolist() == ["x", "y"] and len(pairs) == 3 and pairs["corr"].notna().all()
    capm = stats.regression.capm(xy["y"], xy["x"])
    assert capm["beta"] == pytest.approx(2, abs=0.02) and capm["r2"] > 0.99
    assert stats.risk.beta(xy["y"], xy["x"]) == pytest.approx(2, abs=0.02)
    lag = stats.correlation.lead_lag(xy["y"], xy["x"].shift(-1), max_lag=3)
    assert lag.loc[lag["corr"].idxmax(), "lag"] == 1


def test_partial_correlation(rng):
    f = rng.normal(size=5000)
    df = pd.DataFrame({"a": f + rng.normal(size=5000), "b": f + rng.normal(size=5000), "f": f})
    assert df.corr().loc["a", "b"] > 0.4
    assert abs(stats.correlation.partial(df).loc["a", "b"]) < 0.05


def test_hypothesis(rng):
    shifted = pd.Series(rng.normal(0.5, 1, 500))
    assert stats.hypothesis.mean_zero(shifted)["p_value"] < 0.01
    a, b = pd.Series(rng.normal(size=500)), pd.Series(rng.normal(size=500))
    assert stats.hypothesis.ks_2samp(a, b)["p_value"] > 0.05
    assert (stats.hypothesis.compare(a, b)["p_value"] > 0.01).all()
    ci, samples = stats.hypothesis.bootstrap(shifted, stats.hypothesis.boot_mean, n=500)
    assert ci.ci_low < 0.5 < ci.ci_high and len(samples) == 500


def test_distribution(rng):
    normal = pd.Series(rng.normal(size=2000))
    fat = pd.Series(rng.standard_t(3, size=2000))
    assert (stats.distribution.normality(normal)["p_value"] > 0.01).all()
    assert (stats.distribution.normality(fat)["p_value"] <= 0.01).all()
    assert stats.distribution.fit(fat).iloc[0].distribution == "student_t"
    v = stats.distribution.var(normal).set_index(["level", "method"])
    assert v.loc[(0.95, "parametric"), "VaR"] == pytest.approx(-1.645, abs=0.1)
    assert v.loc[(0.95, "historical"), "CVaR"] < v.loc[(0.95, "historical"), "VaR"]


def test_risk_drawdown():
    idx = pd.bdate_range("2020-01-01", periods=4)
    r = pd.Series([0.0, 0.2, -0.25, 0.5], index=idx)
    mdd = stats.risk.max_drawdown(r)
    assert mdd["max_drawdown"] == pytest.approx(-0.25)
    assert mdd["peak"] == idx[1] and mdd["trough"] == idx[2] and mdd["recovery"] == idx[3]
    summary = stats.risk.summary(pd.DataFrame({"r": r}), bench=r)
    assert summary.loc["r", "tracking_error"] == 0 and summary.loc["r", "beta"] == pytest.approx(1)


def test_timeseries(rng, dates):
    noise = pd.Series(rng.normal(size=len(dates)), index=dates)
    walk = noise.cumsum()
    assert stats.timeseries.adf(noise)["p_value"] < 0.01
    assert stats.timeseries.adf(walk)["p_value"] > 0.05
    long_walk = pd.Series(rng.normal(size=20000)).cumsum()
    assert stats.timeseries.hurst(long_walk) == pytest.approx(0.5, abs=0.05)
    assert stats.timeseries.variance_ratio(noise)["p_value"] > 0.01
    other = walk + rng.normal(size=len(dates))
    assert stats.timeseries.cointegration(other, walk)["p_value"] < 0.01
    assert stats.timeseries.cointegration(walk, pd.Series(rng.normal(size=len(dates)), index=dates).cumsum())[
        "p_value"] > 0.05
    assert stats.timeseries.granger(noise, (noise.shift(-1) + rng.normal(size=len(dates))).rename("lead")).p_value.iloc[0] < 0.01
    assert len(stats.timeseries.autocorr(noise, 10)) == 11
    table, tests = stats.timeseries.seasonality(noise * 0.01)
    assert len(table) == 12 and len(tests) == 2
    assert set(stats.timeseries.decompose(walk, 21).columns) == {"observed", "trend", "seasonal", "resid"}


def test_pca_and_portfolios(rng, dates):
    factor = rng.normal(0, 0.01, len(dates))
    basket = pd.DataFrame({f"s{i}": factor + rng.normal(0, 0.003, len(dates)) for i in range(5)}, index=dates)
    explained, loadings = stats.multivariate.pca(basket)
    assert explained.explained.iloc[0] > 0.8 and (loadings["PC1"] > 0).all()
    table, weights = stats.multivariate.portfolios(basket)
    assert weights.sum().round(6).tolist() == [1, 1]
    assert table.loc["min_variance", "ann_vol"] <= table.loc["equal_weight", "ann_vol"] + 1e-9


def test_risk_cross_section_matches_single_series(rng, dates):
    r = pd.DataFrame({"a": rng.normal(0.001, 0.01, len(dates)), "b": rng.normal(0, 0.02, len(dates))}, index=dates)
    r.iloc[:100, 1] = np.nan  # b starts later
    table = stats.risk.cross_section(r)
    single = stats.risk.metrics(r["b"])
    assert table.loc["b", "observations"] == len(dates) - 100
    assert table.loc["b", "vol_ann"] == pytest.approx(single.ann_vol)
    assert table.loc["b", "max_drawdown"] == pytest.approx(single.max_drawdown)
    assert table.loc["a", "var_95"] == pytest.approx(r["a"].quantile(0.05))
    assert table.loc["a", "cvar_95"] < table.loc["a", "var_95"]


def test_blend_compounds_weighted_component_returns():
    idx = pd.bdate_range("2020-01-01", periods=120)
    rng = np.random.default_rng(4)
    wide = pd.DataFrame({c: 100 * np.cumprod(1 + rng.normal(0.0003, v, len(idx)))
                         for c, v in [("A", 0.012), ("B", 0.004)]}, index=idx)
    blended = returns.blend(wide, {"A": 0.25, "B": 0.75})
    manual = (1 + wide.pct_change().dropna().mul([0.25, 0.75]).sum(axis=1)).cumprod() * 100
    assert blended.iloc[0] == 100
    assert np.allclose(blended.iloc[1:], manual)
    # Weights are normalised, and the blend sits between its components on risk.
    assert np.allclose(returns.blend(wide, {"A": 1, "B": 3}), blended)
    assert blended.pct_change().std() < wide.A.pct_change().std()


def test_matrix_pvalues_match_a_test_per_pair():
    rng = np.random.default_rng(9)
    df = pd.DataFrame(rng.normal(0, 0.01, (400, 6)), columns=list("ABCDEF"))
    df["B"] += df["A"]                 # a genuinely correlated pair
    df.iloc[:80, 2] = np.nan           # a late-starting series
    df.iloc[200:210, 4] = np.nan       # a gap
    for method in ("pearson", "spearman"):
        fast = stats.correlation.pvalues(df, method)
        slow = stats.correlation._per_pair(df, method)
        assert np.allclose(fast.to_numpy(), slow.to_numpy(), atol=1e-10), method
    assert (stats.correlation.pvalues(df).loc["A", "B"] < 0.01)


def test_rolling_average_correlation_tracks_the_matrix_average():
    rng = np.random.default_rng(5)
    factor = rng.normal(0, 0.01, 600)
    r = pd.DataFrame({f"T{i}": 0.6 * factor + rng.normal(0, 0.01, 600) for i in range(8)},
                     index=pd.bdate_range("2022-01-01", periods=600))
    window = 120
    fast = stats.correlation.rolling_average(r, window)
    brute = pd.Series({r.index[i - 1]: r.iloc[i - window:i].corr().to_numpy()[
        np.triu_indices(r.shape[1], 1)].mean() for i in range(window, len(r) + 1)})
    common = fast.index.intersection(brute.index)
    assert len(common) > 400
    assert (fast[common] - brute[common]).abs().mean() < 0.01
    assert fast.between(-1, 1).all()


def test_non_stationary_flags_levels_but_not_returns():
    rng = np.random.default_rng(11)
    idx = pd.bdate_range("2020-01-01", periods=900)
    walk = pd.Series(100 * np.cumprod(1 + rng.normal(0, 0.01, 900)), index=idx)
    frame = pd.DataFrame({"level_a": walk, "level_b": 100 * np.cumprod(1 + rng.normal(0, 0.01, 900)),
                          "returns": walk.pct_change().fillna(0)})
    flagged = stats.timeseries.non_stationary(frame)
    assert set(flagged) == {"level_a", "level_b"}, "price levels drift; returns do not"
    assert stats.timeseries.non_stationary(frame["returns"]) == []


def test_align_closes_recovers_the_cross_market_relationship():
    rng = np.random.default_rng(13)
    idx = pd.bdate_range("2021-01-01", periods=800)
    us = pd.Series(rng.normal(0, 0.01, 800), index=idx)
    # The local market opens after Wall Street closes, so it follows yesterday's US move.
    local = us.shift(1).fillna(0) * 0.8 + rng.normal(0, 0.004, 800)
    prices = pd.DataFrame({"LOCAL": 100 * (1 + local).cumprod(), "US": 100 * (1 + us).cumprod()})
    where = {"LOCAL": "Australia", "US": "United States"}

    raw = prices.pct_change().dropna()
    aligned = returns.align_closes(prices, where).pct_change().dropna()
    assert abs(raw.LOCAL.corr(raw.US)) < 0.2, "same-day pairing hides the link"
    assert aligned.LOCAL.corr(aligned.US) > 0.8, "carrying the US series forward recovers it"
    # Local-only baskets are untouched.
    only_local = returns.align_closes(prices[["LOCAL"]], where)
    assert only_local.equals(prices[["LOCAL"]])


def test_factor_spreads_isolate_the_factor_from_market_direction():
    idx = pd.bdate_range("2022-01-01", periods=300)
    rng = np.random.default_rng(17)
    market = rng.normal(0, 0.01, 300)
    prices = pd.DataFrame({t: 100 * np.cumprod(1 + market + rng.normal(0, 0.002, 300))
                           for t in ["SPY", "IWM", "TLT", "SHY"]}, index=idx)
    built = factors.returns(prices, ["Market (US)", "Size", "Duration"])
    assert list(built.columns) == ["Market (US)", "Size", "Duration"]
    # A spread strips out the common market move that both legs share.
    assert built["Size"].std() < built["Market (US)"].std()
    assert factors.legs(["Size", "Duration"]) == ["IWM", "SPY", "TLT", "SHY"]
    # Factors whose legs are not loaded are skipped rather than erroring.
    assert "Gold" not in factors.returns(prices, ["Market (US)", "Gold"]).columns


def test_tearsheet_records_how_the_numbers_were_made():
    table = pd.DataFrame({"ann_return": [0.08]}, index=["^AXJO"])
    html = tearsheet.build("Report", {"Period": "2021 to 2026", "Adjustments": "Live cash rate"},
                           [("Risk", table), ("Nothing", pd.DataFrame())], notes=["1,264 returns."])
    assert "Live cash rate" in html and "1,264 returns." in html
    assert html.count("<h2>") == 1, "empty sections are left out"
    assert "^AXJO" in html and "Generated" in html


def test_inflation_helpers(monkeypatch):
    quarters = pd.period_range("2020Q1", periods=12, freq="Q").to_timestamp(how="end").normalize()
    cpi = pd.Series(np.linspace(100, 122, 12), index=quarters)   # about 4.6% a year
    monkeypatch.setattr(inflation, "series", lambda region="AUS": cpi)

    rate = inflation.yoy("AUS")
    assert rate.between(0.03, 0.09).all(), "a steady climb gives a steady rate"

    nominal = pd.Series(np.linspace(100, 122, 12), index=quarters)  # grows exactly with prices
    real = inflation.deflate(nominal, "AUS")
    assert real.iloc[0] == pytest.approx(real.iloc[-1], rel=1e-9), "no real growth once prices are removed"
    assert real.iloc[-1] == pytest.approx(nominal.iloc[-1]), "stated in the final period's money"


def test_inflation_hubs_are_configured_with_a_publication_frequency():
    assert set(inflation.HUBS) >= {"AUS", "USA", "GBR", "EA20", "JPN"}
    assert all(freq in ("M", "Q") for _, freq, _ in inflation.HUBS.values())
    assert inflation.HUBS["AUS"][1] == "Q", "the ABS publishes Australian CPI quarterly"
