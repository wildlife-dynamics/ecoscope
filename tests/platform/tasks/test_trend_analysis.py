import numpy as np
import pandas as pd
import pytest
from pydantic import TypeAdapter

from ecoscope.platform.tasks.analysis._trend_analysis import (
    GammFamilySettings,
    GammMcmcSettings,
    GammSplineSettings,
    GammTrendModel,
    GamTrendModel,
    LinearTrendModel,
    TrendModel,
    fit_trend_model,
    get_trend_model_fit_summary,
    is_gamm_trend_model,
    is_not_gamm_trend_model,
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
    gam = adapter.validate_python({"model": "gam", "family_settings": {"family": "poisson"}})
    assert isinstance(gam, GamTrendModel)
    assert gam.family_settings.family == "poisson"


def test_is_gamm_trend_model():
    assert is_gamm_trend_model(GammTrendModel()) is True
    assert is_gamm_trend_model(LinearTrendModel(), GammTrendModel()) is True
    assert is_gamm_trend_model(LinearTrendModel(), GamTrendModel()) is False
    assert is_gamm_trend_model() is False


def test_is_not_gamm_trend_model():
    assert is_not_gamm_trend_model(GammTrendModel()) is False
    assert is_not_gamm_trend_model(LinearTrendModel()) is True


def test_advanced_titled_enum_schema():
    """GammFamilySettings.family (shared by GamTrendModel and
    GammTrendModel) uses _advanced_titled_enum to swap the bare enum for
    labeled oneOf options."""
    schema = GammFamilySettings.model_json_schema()["properties"]["family"]
    assert "enum" not in schema
    assert schema["oneOf"] == [
        {"const": "gaussian", "title": "Gaussian"},
        {"const": "poisson", "title": "Poisson"},
        {"const": "gamma", "title": "Gamma"},
        {"const": "bernoulli", "title": "Bernoulli"},
    ]
    assert schema["ecoscope:advanced"] is True


def test_nullable_advanced_schema():
    """GammMcmcSettings.tune uses _nullable_advanced to rewrite pydantic's
    `anyOf` nullable-number form into the flat array-of-types shorthand this
    project's RJSF/AJV setup needs (see its own docstring)."""
    schema = GammMcmcSettings.model_json_schema()["properties"]["tune"]
    assert "anyOf" not in schema
    assert schema["type"] == ["integer", "null"]
    assert schema["ecoscope:advanced"] is True


def test_fit_and_predict_linear_trend(linear_dataframe):
    model_params = fit_trend_model(linear_dataframe, LinearTrendModel(), time_column="year", value_column="value")
    assert model_params["model"]["model"] == "linear"
    assert model_params["metrics"]["r_squared"] > 0.9

    predictions = predict_trend_model(model_params)
    assert list(predictions.columns) == ["y", "time", "predicted", "ci_lower", "ci_upper"]
    assert len(predictions) == len(linear_dataframe)


def test_fit_and_predict_linear_trend_with_datetime_time_column(linear_dataframe):
    """_prepare_xy converts a datetime `time_column` to numeric (e.g.
    speedmap-trend's `period_start`) before fitting - a plain numeric time
    column like `year` never exercises that conversion."""
    dated_dataframe = linear_dataframe.assign(date=pd.to_datetime(linear_dataframe["year"], format="%Y"))

    model_params = fit_trend_model(dated_dataframe, LinearTrendModel(), time_column="date", value_column="value")
    # Regression check: nanoseconds-since-epoch (the previous unit) makes the
    # OLS design matrix numerically singular once an intercept is added -
    # r_squared silently collapses to near-zero on this otherwise-clean
    # linear series without a well-conditioned unit (see _prepare_xy).
    assert model_params["metrics"]["r_squared"] > 0.9
    assert model_params["time_is_datetime"] is True

    predictions = predict_trend_model(model_params)
    assert len(predictions) == len(dated_dataframe)
    # "time" must round-trip back to real timestamps, not the raw
    # days-since-epoch numbers _prepare_xy fits on internally.
    assert pd.api.types.is_datetime64_any_dtype(predictions["time"])
    pd.testing.assert_series_equal(predictions["time"], dated_dataframe["date"], check_names=False, check_freq=False)


def test_fit_and_predict_gam_trend(linear_dataframe):
    """GAM is now Bayesian (PyMC/Bambi), like GAMM - no frequentist aic/bic,
    and no per-site pooling (see GamTrendModel's docstring)."""
    pytest.importorskip("bambi")
    model_params = fit_trend_model(
        linear_dataframe,
        GamTrendModel(
            spline_settings=GammSplineSettings(degree_of_freedom=5),
            mcmc_settings=GammMcmcSettings(draws=100, tune=100, chains=1, random_seed=42),
        ),
        time_column="year",
        value_column="value",
    )
    assert model_params["model"]["model"] == "gam"
    assert "aic" not in model_params["metrics"]  # GAM is Bayesian; aic/bic don't apply.

    predictions = predict_trend_model(model_params, time_values=[2000.0, 2010.0])
    assert list(predictions["time"]) == [2000.0, 2010.0]
    assert predictions["y"].isna().all()


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


def test_fit_trend_model_tags_name_for_single_group_only(multi_site_dataframe):
    """fit_trend_model's "name" field identifies a single-group fit (for
    get_trend_model_fit_summary to tag its output with) - None for a fit
    spanning more than one group, like GAMM's own combined-across-groups fit."""
    site_a = multi_site_dataframe[multi_site_dataframe["name"] == "Site A"]
    single_group_params = fit_trend_model(site_a, LinearTrendModel(), time_column="year", value_column="value")
    assert single_group_params["name"] == "Site A"

    combined_params = fit_trend_model(
        multi_site_dataframe,
        GammTrendModel(
            spline_settings=GammSplineSettings(degree_of_freedom=3),
            mcmc_settings=GammMcmcSettings(draws=100, tune=100, chains=1),
        ),
        time_column="year",
        value_column="value",
    )
    assert combined_params["name"] is None


def test_get_trend_model_fit_summary_tags_name_for_single_group(multi_site_dataframe):
    """Per-group fit summaries (e.g. GAM/Linear, each fit independently per
    site) get a leading "name" column so they stay identifiable once
    concatenated into one combined table - unlike GAMM's combined-across-
    groups summary, which has no single name to attribute rows to."""
    site_a = multi_site_dataframe[multi_site_dataframe["name"] == "Site A"]
    model = LinearTrendModel()
    model_params = fit_trend_model(site_a, model, time_column="year", value_column="value")
    summary = get_trend_model_fit_summary(model_params, model=model)

    assert list(summary.columns)[0] == "name"
    assert (summary["name"] == "Site A").all()

    combined_model = GammTrendModel(
        spline_settings=GammSplineSettings(degree_of_freedom=3),
        mcmc_settings=GammMcmcSettings(draws=100, tune=100, chains=1),
    )
    combined_params = fit_trend_model(multi_site_dataframe, combined_model, time_column="year", value_column="value")
    combined_summary = get_trend_model_fit_summary(combined_params, model=combined_model)
    assert "name" not in combined_summary.columns


def test_get_trend_model_fit_summary_dataframe_param_does_not_change_content(multi_site_dataframe):
    """The optional `dataframe` param exists only so a workflow can map this
    task over every group's own dataframe (mirroring predict_trend_model) -
    e.g. to show a combined-across-groups fit's one summary under every
    group's own dashboard view. It must not change the computed summary."""
    model = GammTrendModel(
        spline_settings=GammSplineSettings(degree_of_freedom=3),
        mcmc_settings=GammMcmcSettings(draws=100, tune=100, chains=1, random_seed=42),
    )
    model_params = fit_trend_model(multi_site_dataframe, model, time_column="year", value_column="value")

    summary_without_dataframe = get_trend_model_fit_summary(model_params, model=model)
    site_b = multi_site_dataframe[multi_site_dataframe["name"] == "Site B"]
    summary_with_dataframe = get_trend_model_fit_summary(model_params, model=model, dataframe=site_b)

    pd.testing.assert_frame_equal(summary_without_dataframe, summary_with_dataframe)


def test_get_trend_model_fit_summary_linear_model(linear_dataframe):
    """Linear fits via statsmodels - one row per coefficient, with a
    p-value/confidence interval (frequentist statistics), unlike GAM/GAMM's
    Bayesian posterior summary below."""
    model = LinearTrendModel()
    model_params = fit_trend_model(linear_dataframe, model, time_column="year", value_column="value")
    summary = get_trend_model_fit_summary(model_params, model=model)

    assert len(summary) > 0
    assert list(summary.columns) == ["parameter", "coefficient", "std_error", "p_value", "ci_lower", "ci_upper"]
    assert summary["ci_lower"].le(summary["ci_upper"]).all()


@pytest.mark.parametrize(
    "model",
    [
        GamTrendModel(
            spline_settings=GammSplineSettings(degree_of_freedom=3),
            mcmc_settings=GammMcmcSettings(draws=100, tune=100, chains=1),
        ),
        GammTrendModel(
            spline_settings=GammSplineSettings(degree_of_freedom=3),
            mcmc_settings=GammMcmcSettings(draws=100, tune=100, chains=1),
        ),
    ],
    ids=["gam", "gamm"],
)
def test_get_trend_model_fit_summary_bayesian_models(linear_dataframe, model):
    """GAM and GAMM both fit via PyMC/Bambi - a posterior summary (mean/sd/
    hdi/r_hat), not the frequentist coefficient/p-value/CI shape above."""
    pytest.importorskip("bambi")
    model_params = fit_trend_model(linear_dataframe, model, time_column="year", value_column="value")
    summary = get_trend_model_fit_summary(model_params, model=model)

    assert len(summary) > 0
    assert "parameter" in summary.columns
    assert "mean" in summary.columns
    assert "r_hat" in summary.columns
