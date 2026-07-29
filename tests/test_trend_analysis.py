"""Tests for trend analysis module."""

import numpy as np
import pytest

from ecoscope.analysis import trend_analysis


@pytest.fixture
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


def test_gamm_regressor_aic_bic_raise(fitted_gamm):
    """GAMM has no frequentist AIC/BIC — point callers at waic/loo."""
    gamm, *_ = fitted_gamm
    with pytest.raises(NotImplementedError, match="waic"):
        gamm.aic()
    with pytest.raises(NotImplementedError, match="loo"):
        gamm.bic()


def test_gamm_regressor_invalid_inference_method():
    """inference_method is validated at init."""
    with pytest.raises(ValueError, match="inference_method"):
        trend_analysis.GAMMRegressor(inference_method="nuts")


def test_gam_regressor_fit_predict(sample_data):
    """Test GAMRegressor fit and predict with auto-optimized alpha."""
    X, y = sample_data
    gam = trend_analysis.GAMRegressor()
    gam.fit(X, y)
    assert gam.alpha_ > 0
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


@pytest.mark.parametrize("family", ["gaussian", "poisson", "gamma", "binomial"])
def test_glm_regressor_fit_predict_per_family(family, sample_data):
    """Basic GLM fit/predict coverage per family."""
    X, y = sample_data
    if family == "binomial":
        y = np.linspace(0.9, 0.1, len(X))
    else:
        y = np.clip(y, 1.0, None)

    model = trend_analysis.GLMRegressor(family=family).fit(X, y)
    predictions = model.predict(X)
    assert len(predictions) == len(y)
    assert all(np.isfinite(predictions))
    if family != "binomial":
        assert all(predictions >= 0)
    if family in ("gaussian", "poisson", "gamma"):
        assert predictions.min() < float(y.mean())


def test_gam_regressor_predict_without_fit():
    """Test that predict raises error if model not fitted."""
    gam = trend_analysis.GAMRegressor(alpha=0.1)
    X = np.array([2000, 2001, 2002])
    with pytest.raises(ValueError, match="Model has not been fitted"):
        gam.predict(X)


def test_gam_regressor_aic_without_fit():
    """Test that aic raises error if model not fitted."""
    gam = trend_analysis.GAMRegressor(alpha=0.1)
    with pytest.raises(ValueError, match="Model has not been fitted"):
        gam.aic()


def test_gam_regressor_bic_without_fit():
    """Test that bic raises error if model not fitted."""
    gam = trend_analysis.GAMRegressor(alpha=0.1)
    with pytest.raises(ValueError, match="Model has not been fitted"):
        gam.bic()


def test_gam_regressor_predict_with_ci_without_fit():
    """Test that predict_with_ci raises error if model not fitted."""
    gam = trend_analysis.GAMRegressor(alpha=0.1)
    X = np.array([2000, 2001, 2002])
    with pytest.raises(ValueError, match="Model has not been fitted"):
        gam.predict_with_ci(X)


def test_gam_regressor_predict_with_ci(sample_data):
    """Test GAMRegressor prediction with confidence intervals."""
    X, y = sample_data
    gam = trend_analysis.GAMRegressor(alpha=0.1).fit(X, y)
    mean, ci_lower, ci_upper = gam.predict_with_ci(X)
    assert len(mean) == len(y)
    assert len(ci_lower) == len(y)
    assert len(ci_upper) == len(y)
    valid_mask = np.isfinite(mean) & np.isfinite(ci_lower) & np.isfinite(ci_upper)
    if valid_mask.any():
        assert all(ci_lower[valid_mask] <= mean[valid_mask])
        assert all(mean[valid_mask] <= ci_upper[valid_mask])


def test_gam_regressor_aic_bic(sample_data):
    """Test GAMRegressor AIC and BIC methods."""
    X, y = sample_data
    gam = trend_analysis.GAMRegressor(alpha=0.1).fit(X, y)
    aic = gam.aic()
    bic = gam.bic()
    assert isinstance(aic, (int, float))
    assert isinstance(bic, (int, float))
    assert aic is not None
    assert bic is not None


def test_gam_regressor_mse(sample_data):
    """Test GAMRegressor MSE method."""
    X, y = sample_data
    gam = trend_analysis.GAMRegressor(alpha=0.1).fit(X, y)
    mse = gam.mse(X, y)
    assert isinstance(mse, float)
    assert mse >= 0


def test_gam_regressor_mse_without_fit():
    """Test that mse raises error if model not fitted."""
    gam = trend_analysis.GAMRegressor(alpha=0.1)
    X = np.array([2000, 2001, 2002])
    y = np.array([100, 95, 90])
    with pytest.raises(ValueError, match="Model has not been fitted"):
        gam.mse(X, y)


def test_gam_regressor_r_squared(sample_data):
    """Test GAMRegressor R-squared method."""
    X, y = sample_data
    gam = trend_analysis.GAMRegressor(alpha=0.1).fit(X, y)
    r2 = gam.r_squared(X, y)
    assert isinstance(r2, float)
    assert r2 <= 1.0


def test_gam_regressor_r_squared_without_fit():
    """Test that r_squared raises error if model not fitted."""
    gam = trend_analysis.GAMRegressor(alpha=0.1)
    X = np.array([2000, 2001, 2002])
    y = np.array([100, 95, 90])
    with pytest.raises(ValueError, match="Model has not been fitted"):
        gam.r_squared(X, y)


def test_gam_regressor_custom_parameters(sample_data):
    """Test GAMRegressor with custom degree_of_freedom, degree, and family."""
    X, y = sample_data
    gam = trend_analysis.GAMRegressor(alpha=0.5, degree_of_freedom=15, degree=2).fit(X, y)
    predictions = gam.predict(X)
    assert len(predictions) == len(y)
    assert all(np.isfinite(predictions))


def test_gam_regressor_poisson_family():
    """Test GAMRegressor with Poisson family."""
    X = np.array([2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009])
    y = np.array([100, 95, 90, 85, 80, 75, 70, 65, 60, 55])
    gam = trend_analysis.GAMRegressor(alpha=0.1, family="poisson").fit(X, y)
    predictions = gam.predict(X)
    assert len(predictions) == len(y)
    assert all(np.isfinite(predictions))
    assert all(predictions >= 0)


def test_gam_regressor_invalid_family():
    """Test GAMRegressor raises error for invalid family."""
    with pytest.raises(ValueError, match="Unsupported family"):
        trend_analysis.GAMRegressor(family="invalid")


def test_gam_regressor_auto_optimize(sample_data):
    """Test GAMRegressor auto-optimizes alpha when not specified."""
    X, y = sample_data
    gam = trend_analysis.GAMRegressor().fit(X, y, metric="aic")
    assert gam.alpha_ > 0
    predictions = gam.predict(X)
    assert len(predictions) == len(y)


def test_gam_regressor_auto_optimize_metrics(sample_data):
    """Test GAMRegressor auto-optimization with all supported metrics."""
    X, y = sample_data
    for metric in ("aic", "bic", "euclidean", "mse", "r_squared"):
        gam = trend_analysis.GAMRegressor().fit(X, y, metric=metric)
        assert gam.alpha_ > 0


def test_gam_regressor_auto_optimize_custom_params(sample_data):
    """Test GAMRegressor auto-optimization with custom model parameters."""
    X, y = sample_data
    gam = trend_analysis.GAMRegressor(degree_of_freedom=15, degree=2).fit(X, y, metric="aic")
    assert gam.alpha_ > 0


def test_gam_regressor_auto_optimize_bound_padding(sample_data):
    """Test GAMRegressor auto-optimization with custom bound_padding_ratio."""
    X, y = sample_data
    gam1 = trend_analysis.GAMRegressor().fit(X, y, bound_padding_ratio=0.1)
    gam2 = trend_analysis.GAMRegressor().fit(X, y, bound_padding_ratio=0.2)
    assert gam1.alpha_ > 0
    assert gam2.alpha_ > 0


def test_gam_regressor_auto_optimize_explicit_bounds(sample_data):
    """Test GAMRegressor auto-optimization with explicit bounds."""
    X, y = sample_data
    gam = trend_analysis.GAMRegressor().fit(X, y, lower_bound=1995.0, upper_bound=2015.0)
    assert gam.alpha_ > 0
