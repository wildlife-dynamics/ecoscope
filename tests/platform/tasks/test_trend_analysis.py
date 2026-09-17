import numpy as np
import pandas as pd
import pytest
from pydantic import TypeAdapter

from ecoscope.platform.tasks.analysis._trend_analysis import (
    GammTrendModel,
    GamTrendModel,
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


@pytest.fixture(scope="module")
def site_dataframe():
    years = np.tile(np.arange(2000, 2010), 3)
    sites = np.repeat(["a", "b", "c"], 10)
    rng = np.random.default_rng(1)
    values = 100 - 1.5 * (years - 2000) + rng.normal(scale=0.5, size=len(years))
    return pd.DataFrame({"year": years, "value": values, "site": sites})


def test_set_trend_model_default():
    model = set_trend_model(GamTrendModel())
    assert isinstance(model, GamTrendModel)
    assert model.model == "gam"


def test_trend_model_discriminated_union_validates_by_model_field():
    adapter: TypeAdapter = TypeAdapter(TrendModel)
    assert isinstance(adapter.validate_python({"model": "linear"}), LinearTrendModel)
    assert isinstance(adapter.validate_python({"model": "glm", "family": "poisson"}), GlmTrendModel)


def test_fit_and_predict_linear_trend(linear_dataframe):
    model_params = fit_trend_model(linear_dataframe, LinearTrendModel(), time_column="year", value_column="value")
    assert model_params["model"]["model"] == "linear"
    assert model_params["metrics"]["r_squared"] > 0.9

    predictions = predict_trend_model(model_params)
    assert list(predictions.columns) == ["y", "time", "predicted", "ci_lower", "ci_upper"]
    assert len(predictions) == len(linear_dataframe)


def test_fit_and_predict_glm_trend(linear_dataframe):
    model_params = fit_trend_model(
        linear_dataframe, GlmTrendModel(family="gaussian"), time_column="year", value_column="value"
    )
    predictions = predict_trend_model(model_params, include_ci=False)
    assert list(predictions.columns) == ["y", "time", "predicted"]


def test_fit_and_predict_gam_trend(linear_dataframe):
    model_params = fit_trend_model(
        linear_dataframe,
        GamTrendModel(alpha=1.0, degree_of_freedom=5, degree=2),
        time_column="year",
        value_column="value",
    )
    assert model_params["metrics"]["aic"] is not None

    predictions = predict_trend_model(model_params, time_values=[2000.0, 2010.0])
    assert list(predictions["time"]) == [2000.0, 2010.0]
    assert predictions["y"].isna().all()


def test_fit_trend_model_requires_site_id_column(linear_dataframe):
    with pytest.raises(ValueError, match="site_id_column"):
        fit_trend_model(
            linear_dataframe,
            GammTrendModel(site_id_column="missing_column"),
            time_column="year",
            value_column="value",
        )


def test_fit_and_predict_gamm_trend(site_dataframe):
    model_params = fit_trend_model(
        site_dataframe,
        GammTrendModel(site_id_column="site", degree_of_freedom=3, draws=100, tune=100, chains=1),
        time_column="year",
        value_column="value",
    )
    assert model_params["model"]["model"] == "gamm"
    assert model_params["site_ids"] is not None
    assert "aic" not in model_params["metrics"]  # GAMM is Bayesian; aic/bic don't apply.

    predictions = predict_trend_model(model_params)
    assert len(predictions) == len(site_dataframe)
