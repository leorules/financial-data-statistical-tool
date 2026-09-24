"""Optional adjustments: off by default, and each one moves the result in the direction it claims."""
import numpy as np
import pandas as pd
import pytest

from market import adjust, cash
from market.config import RISK_FREE
from market.stats import correlation, multivariate, regression, risk


@pytest.fixture
def series():
    rng = np.random.default_rng(7)
    index = pd.bdate_range("2015-01-01", periods=750)
    bench = pd.Series(rng.normal(0.0004, 0.01, len(index)), index=index, name="bench")
    return bench + pd.Series(rng.normal(0.0002, 0.006, len(index)), index=index, name="asset"), bench


@pytest.fixture
def panel():
    rng = np.random.default_rng(11)
    index = pd.bdate_range("2018-01-01", periods=300)
    factor = rng.normal(0, 0.01, len(index))
    return pd.DataFrame({f"A{i}": factor * (0.3 + 0.1 * i) + rng.normal(0, 0.008, len(index))
                         for i in range(6)}, index=index)


# --- defaults are the unadjusted path ---------------------------------------------------------------

def test_defaults_match_the_unadjusted_calculation(series, panel):
    asset, bench = series
    assert risk.metrics(asset, bench).equals(risk.metrics(asset, bench, rf=RISK_FREE))
    assert regression.capm(asset, bench).equals(regression.capm(asset, bench, hac=False))
    assert correlation.pvalues(panel).equals(correlation.pvalues(panel, adjust=False))
    table, weights = multivariate.portfolios(panel)
    plain_table, plain_weights = multivariate.portfolios(panel, shrink=False)
    assert table.equals(plain_table) and weights.equals(plain_weights)


def test_flat_rate_sharpe_is_unchanged_by_an_equivalent_series(series):
    asset, _ = series
    flat = pd.Series(RISK_FREE, index=asset.index)
    assert risk.metrics(asset).sharpe == pytest.approx(risk.metrics(asset, rf=flat).sharpe)


# --- live cash rate ---------------------------------------------------------------------------------

def test_a_lower_cash_rate_raises_the_sharpe_ratio(series):
    asset, _ = series
    cheap = pd.Series(0.001, index=asset.index)
    assert risk.metrics(asset, rf=cheap).sharpe > risk.metrics(asset).sharpe


def test_cash_rate_falls_back_where_no_data_covers_the_dates(monkeypatch):
    index = pd.bdate_range("1975-01-01", periods=10)
    monkeypatch.setattr(cash, "_yield_series", lambda currency: pd.Series(dtype=float))
    rates = cash.annual_rate(index, "AUD")
    assert (rates == RISK_FREE).all()
    assert "flat" in cash.label(index, "AUD")


# --- robust standard errors -------------------------------------------------------------------------

def test_hac_keeps_coefficients_and_only_moves_inference(series):
    asset, bench = series
    plain, robust = regression.capm(asset, bench), regression.capm(asset, bench, hac=True)
    assert plain.beta == pytest.approx(robust.beta)
    assert plain.beta_p != robust.beta_p


# --- false-discovery correction ---------------------------------------------------------------------

def test_fdr_never_lowers_a_p_value(panel):
    raw = correlation.pvalues(panel)
    adjusted = correlation.pvalues(panel, adjust=True)
    upper = np.triu_indices(len(raw), k=1)
    assert (adjusted.to_numpy()[upper] >= raw.to_numpy()[upper] - 1e-12).all()
    assert (adjusted.to_numpy()[upper] > raw.to_numpy()[upper]).any()


# --- covariance shrinkage ---------------------------------------------------------------------------

def test_shrinkage_conditions_a_covariance_that_has_too_few_observations():
    rng = np.random.default_rng(3)
    wide = pd.DataFrame(rng.normal(0, 0.01, (12, 20)), index=pd.bdate_range("2020-01-01", periods=12))
    sample = np.linalg.eigvalsh(multivariate.covariance(wide))
    shrunk = np.linalg.eigvalsh(multivariate.covariance(wide, shrink=True))
    assert sample.min() < 1e-12 <= shrunk.min()
    assert 0 < multivariate.shrinkage_intensity(wide) <= 1


# --- registry ---------------------------------------------------------------------------------------

def test_every_adjustment_is_labelled_and_explained():
    assert set(adjust.ADJUSTMENTS) == {"total_return", "live_cash", "align_closes", "hac", "fdr", "shrinkage"}
    assert all(label and why for label, why in adjust.ADJUSTMENTS.values())
    assert adjust.active({"hac", "fdr"}, "fdr") == ["False-discovery correction"]
    assert adjust.active(frozenset()) == []
