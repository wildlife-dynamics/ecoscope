import numpy as np
import pandas as pd
import pytest
from pydantic import TypeAdapter

from ecoscope.platform.tasks.analysis._trend_analysis import (
    GammMcmcSettings,
    GammSplineSettings,
    GammTrendModel,
    GamSmoothingSettings,
    GamSplineSettings,
    GamTrendModel,
    GlmFamilySettings,
    GlmTrendModel,
    LinearTrendModel,
    TrendModel,
    fit_trend_model,
    predict_trend_model,
    set_trend_model,
)


@pytest.fixture(scope="module")
def linear_dataframe():
    years = np.arange(2000, 2020)
    rng = np.random.default_rng(0)
    values = 100 - 2.0 * (years - 2000) + rng.normal(scale=0.5, size=len(years))
    return pd.DataFrame({"year": years, "value": values})


def test_set_trend_model_default():
    model = set_trend_model(GamTrendModel())
    assert isinstance(model, GamTrendModel)
    assert model.model == "gam"


def test_trend_model_discriminated_union_validates_by_model_field():
    adapter: TypeAdapter = TypeAdapter(TrendModel)
    assert isinstance(adapter.validate_python({"model": "linear"}), LinearTrendModel)
    glm = adapter.validate_python({"model": "glm", "family_settings": {"family": "poisson"}})
    assert isinstance(glm, GlmTrendModel)
    assert glm.family_settings.family == "poisson"


def test_fit_and_predict_linear_trend(linear_dataframe):
    model_params = fit_trend_model(linear_dataframe, LinearTrendModel(), time_column="year", value_column="value")
    assert model_params["model"]["model"] == "linear"
    assert model_params["metrics"]["r_squared"] > 0.9

    predictions = predict_trend_model(model_params)
    assert list(predictions.columns) == ["y", "time", "predicted", "ci_lower", "ci_upper"]
    assert len(predictions) == len(linear_dataframe)


def test_fit_and_predict_glm_trend(linear_dataframe):
    model_params = fit_trend_model(
        linear_dataframe,
        GlmTrendModel(family_settings=GlmFamilySettings(family="gaussian")),
        time_column="year",
        value_column="value",
    )
    predictions = predict_trend_model(model_params, include_ci=False)
    assert list(predictions.columns) == ["y", "time", "predicted"]


def test_fit_and_predict_gam_trend(linear_dataframe):
    model_params = fit_trend_model(
        linear_dataframe,
        GamTrendModel(
            smoothing_settings=GamSmoothingSettings(alpha=1.0),
            spline_settings=GamSplineSettings(degree_of_freedom=5, degree=2),
        ),
        time_column="year",
        value_column="value",
    )
    assert model_params["metrics"]["aic"] is not None

    predictions = predict_trend_model(model_params, time_values=[2000.0, 2010.0])
    assert list(predictions["time"]) == [2000.0, 2010.0]
    assert predictions["y"].isna().all()


@pytest.mark.parametrize("alpha", ["", None])
def test_gam_alpha_left_empty_selects_automatically(linear_dataframe, alpha):
    """alpha is `float | None` with a `_empty_string_to_none` BeforeValidator
    (see GamSmoothingSettings) - a defensive backstop in case "" ever
    reaches this code, alongside the actual fix for RJSF/AJV rejecting an
    empty field (json_schema_extra=_nullable_advanced, which compiles to
    the flat `type: ["number", "null"]` shorthand rather than pydantic's
    own `anyOf` form). Both "" and None must mean "leave empty, select
    automatically"."""
    model_params = fit_trend_model(
        linear_dataframe,
        GamTrendModel(smoothing_settings=GamSmoothingSettings(alpha=alpha)),
        time_column="year",
        value_column="value",
    )
    assert model_params["metrics"]["aic"] is not None


def test_gam_alpha_accepts_numeric_string(linear_dataframe):
    """A filled-in RJSF number field can submit a numeric string too - the
    _empty_string_to_none BeforeValidator normalizes it to a real float."""
    model = GamTrendModel(smoothing_settings=GamSmoothingSettings(alpha="1.5"))
    assert model.smoothing_settings.alpha == 1.5

    model_params = fit_trend_model(linear_dataframe, model, time_column="year", value_column="value")
    assert model_params["model"]["smoothing_settings"]["alpha"] == 1.5


def test_fit_and_predict_gamm_trend_without_name_column_uses_constant_site(linear_dataframe):
    """No "name" column (e.g. a caller outside this workflow) falls back to
    a single constant site rather than raising - GAMM's random effect is
    then just a degenerate single-group intercept."""
    model_params = fit_trend_model(
        linear_dataframe,
        GammTrendModel(
            spline_settings=GammSplineSettings(degree_of_freedom=3),
            mcmc_settings=GammMcmcSettings(draws=100, tune=100, chains=1),
        ),
        time_column="year",
        value_column="value",
    )
    assert model_params["model"]["model"] == "gamm"
    assert model_params["site_ids"] == [0.0] * len(linear_dataframe)
    assert "aic" not in model_params["metrics"]  # GAMM is Bayesian; aic/bic don't apply.

    predictions = predict_trend_model(model_params)
    assert len(predictions) == len(linear_dataframe)


@pytest.fixture(scope="module")
def multi_site_dataframe():
    years = np.tile(np.arange(2000, 2010), 3)
    names = np.repeat(["Site A", "Site B", "Site C"], 10)
    rng = np.random.default_rng(1)
    # Deliberately different intercepts per site so a real random effect
    # should give each site a visibly different group-specific prediction.
    offsets = np.repeat([0.0, 40.0, -40.0], 10)
    values = 100 - 1.5 * (years - 2000) + offsets + rng.normal(scale=0.5, size=len(years))
    return pd.DataFrame({"year": years, "value": values, "name": names})


def test_fit_gamm_combined_across_sites_uses_real_site_variance(multi_site_dataframe):
    """Fit once across every site's combined data (see GammTrendModel's
    docstring) - site_ids carries real cross-site variance, not a single
    repeated value like the degenerate case above."""
    model_params = fit_trend_model(
        multi_site_dataframe,
        GammTrendModel(
            spline_settings=GammSplineSettings(degree_of_freedom=3),
            mcmc_settings=GammMcmcSettings(draws=100, tune=100, chains=1),
        ),
        time_column="year",
        value_column="value",
    )
    assert set(model_params["site_ids"]) == {"Site A", "Site B", "Site C"}


def test_predict_gamm_from_combined_fit_gives_each_site_its_own_curve(multi_site_dataframe):
    """predict_trend_model's `dataframe` parameter gets that site's own
    group-specific prediction out of a fit combined across multiple sites -
    the whole point of a real (non-degenerate) random effect."""
    model_params = fit_trend_model(
        multi_site_dataframe,
        GammTrendModel(
            spline_settings=GammSplineSettings(degree_of_freedom=3),
            mcmc_settings=GammMcmcSettings(draws=200, tune=200, chains=1, random_seed=0),
        ),
        time_column="year",
        value_column="value",
    )

    site_a = multi_site_dataframe[multi_site_dataframe["name"] == "Site A"]
    site_b = multi_site_dataframe[multi_site_dataframe["name"] == "Site B"]

    predictions_a = predict_trend_model(model_params, dataframe=site_a, time_column="year", value_column="value")
    predictions_b = predict_trend_model(model_params, dataframe=site_b, time_column="year", value_column="value")

    assert list(predictions_a["y"]) == list(site_a["value"])  # observed values pass through
    # Site B was generated ~80 units higher than Site A - its group-specific
    # prediction should reflect that, not collapse to the same population curve.
    assert (predictions_b["predicted"].mean() - predictions_a["predicted"].mean()) > 30
