"""Tests for trend analysis module."""

import numpy as np
import pytest

from ecoscope.analysis import trend_analysis


@pytest.fixture(scope="module")
def sample_data():
    """Create sample time series data with some noise to avoid perfect separation."""
    X = np.array([2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009])
    # Add some noise to avoid perfect separation
    np.random.seed(42)
    y = np.array([100, 95, 90, 85, 80, 75, 70, 65, 60, 55]) + np.random.normal(0, 2, len(X))
    return X, y


@pytest.fixture(scope="module")
def gamm_sample_data():
    """Multi-site panel for GAMMRegressor tests."""
    np.random.seed(42)
    sites = ["A", "B"]
    X_list, y_list, sid_list = [], [], []
    for i, site in enumerate(sites):
        years = np.arange(2000, 2010)
        y = 100 - (years - 2000) * 2 + np.random.normal(0, 1, len(years))
        y += 10 * i  # site B sits higher on average
        X_list.extend(years)
        y_list.extend(y)
        sid_list.extend([site] * len(years))
    return np.array(X_list), np.array(y_list), np.array(sid_list)


@pytest.fixture(scope="module")
def fitted_gamm(gamm_sample_data):
    """Shared fitted GAMM for tests (MCMC with minimal draws for speed)."""
    pytest.importorskip("bambi")
    X, y, sites = gamm_sample_data
    gamm = trend_analysis.GAMMRegressor(
        degree_of_freedom=5,
        inference_method="mcmc",
        draws=100,
        tune=100,
        chains=1,
        random_seed=42,
    )
    gamm.fit(X, y, sites)
    return gamm, X, y, sites


def test_gamm_regressor_fit_predict(fitted_gamm):
    """Test GAMMRegressor fit and site-specific predict."""
    gamm, X, y, sites = fitted_gamm
    predictions = gamm.predict(X, site_ids=sites)
    assert len(predictions) == len(y)
    assert all(np.isfinite(predictions))


def test_gamm_regressor_predict_without_fit():
    """Test that predict raises error if GAMM model not fitted."""
    pytest.importorskip("bambi")
    gamm = trend_analysis.GAMMRegressor(degree_of_freedom=5)
    X = np.array([2000, 2001, 2002])
    with pytest.raises(ValueError, match="Model has not been fitted"):
        gamm.predict(X)


def test_gamm_regressor_predict_with_ci(fitted_gamm):
    """Test GAMMRegressor prediction with credible intervals."""
    gamm, X, y, sites = fitted_gamm
    mean, ci_lower, ci_upper = gamm.predict_with_ci(X, site_ids=sites)
    assert len(mean) == len(y)
    assert len(ci_lower) == len(y)
    assert len(ci_upper) == len(y)
    assert all(ci_lower <= mean)
    assert all(mean <= ci_upper)


def test_gamm_regressor_population_predict(fitted_gamm):
    """Population predict zeros site random effects; site-specific predict does not."""
    gamm, _X, _y, _sites = fitted_gamm

    X_one = np.array([2005.0])
    pop_pred = gamm.predict(X_one, site_ids=None)
    site_b_pred = gamm.predict(X_one, site_ids=np.array(["B"]))

    assert pop_pred.shape == (1,)
    assert site_b_pred.shape == (1,)
    assert site_b_pred[0] > pop_pred[0]


def test_gamm_regressor_r_squared_and_mse(fitted_gamm):
    """Test GAMMRegressor r_squared and mse."""
    gamm, X, y, sites = fitted_gamm
    r2 = gamm.r_squared(X, y, site_ids=sites)
    mse = gamm.mse(X, y, site_ids=sites)
    assert isinstance(r2, float)
    assert isinstance(mse, float)
    assert mse >= 0
    assert r2 <= 1.0


def test_gamm_regressor_loo(fitted_gamm):
    """GAMM's Bayesian fit exposes ArviZ LOO (PSIS-LOO-CV) in place of aic/bic -
    requires compute_log_likelihood() to have been called during fit()."""
    gamm, *_ = fitted_gamm
    loo = gamm.loo()
    assert loo is not None


def test_gamm_regressor_aic_bic_raise(fitted_gamm):
    """GAMM has no frequentist AIC/BIC — point callers at loo() (ArviZ 1.0+
    dropped waic() outright in favor of PSIS-LOO)."""
    gamm, *_ = fitted_gamm
    with pytest.raises(NotImplementedError, match="loo"):
        gamm.aic()
    with pytest.raises(NotImplementedError, match="loo"):
        gamm.bic()


def test_gamm_regressor_invalid_inference_method():
    """inference_method is validated at init."""
    with pytest.raises(ValueError, match="inference_method"):
        trend_analysis.GAMMRegressor(inference_method="nuts")


@pytest.fixture(scope="module")
def fitted_gam(sample_data):
    """Shared fitted GAM for tests (MCMC with minimal draws for speed)."""
    pytest.importorskip("bambi")
    X, y = sample_data
    gam = trend_analysis.GAMRegressor(
        degree_of_freedom=5,
        inference_method="mcmc",
        draws=100,
        tune=100,
        chains=1,
        random_seed=42,
    )
    gam.fit(X, y)
    return gam, X, y


def test_gam_regressor_fit_predict(fitted_gam):
    """Test GAMRegressor fit and predict."""
    gam, X, y = fitted_gam
    predictions = gam.predict(X)
    assert len(predictions) == len(y)
    assert all(np.isfinite(predictions))


def test_linear_regression_fit_predict(sample_data):
    """Test LinearRegressionRegressor fit and predict."""
    X, y = sample_data
    model = trend_analysis.LinearRegressionRegressor().fit(X, y)
    predictions = model.predict(X)
    assert len(predictions) == len(y)
    assert all(np.isfinite(predictions))


def test_gam_regressor_predict_without_fit():
    """Test that predict raises error if GAM model not fitted."""
    pytest.importorskip("bambi")
    gam = trend_analysis.GAMRegressor(degree_of_freedom=5)
    X = np.array([2000, 2001, 2002])
    with pytest.raises(ValueError, match="Model has not been fitted"):
        gam.predict(X)


def test_gam_regressor_predict_with_ci(fitted_gam):
    """Test GAMRegressor prediction with credible intervals."""
    gam, X, y = fitted_gam
    mean, ci_lower, ci_upper = gam.predict_with_ci(X)
    assert len(mean) == len(y)
    assert len(ci_lower) == len(y)
    assert len(ci_upper) == len(y)
    assert all(ci_lower <= mean)
    assert all(mean <= ci_upper)


def test_gam_regressor_r_squared_and_mse(fitted_gam):
    """Test GAMRegressor r_squared and mse."""
    gam, X, y = fitted_gam
    r2 = gam.r_squared(X, y)
    mse = gam.mse(X, y)
    assert isinstance(r2, float)
    assert isinstance(mse, float)
    assert mse >= 0
    assert r2 <= 1.0


def test_gam_regressor_aic_bic_raise(fitted_gam):
    """GAM has no frequentist AIC/BIC — point callers at loo() (ArviZ 1.0+
    dropped waic() outright in favor of PSIS-LOO)."""
    gam, *_ = fitted_gam
    with pytest.raises(NotImplementedError, match="loo"):
        gam.aic()
    with pytest.raises(NotImplementedError, match="loo"):
        gam.bic()


def test_gam_regressor_loo(fitted_gam):
    """GAM's Bayesian fit exposes ArviZ LOO (PSIS-LOO-CV) in place of aic/bic."""
    gam, *_ = fitted_gam
    loo = gam.loo()
    assert loo is not None


def test_gam_regressor_invalid_inference_method():
    """inference_method is validated at init."""
    with pytest.raises(ValueError, match="inference_method"):
        trend_analysis.GAMRegressor(inference_method="nuts")
