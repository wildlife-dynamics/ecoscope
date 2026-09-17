from typing import Annotated, Literal, TypeAlias, cast

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field
from wt_registry import register

from ecoscope.platform.annotations import AdvancedField, AnyDataFrame

# `ecoscope.analysis.trend_analysis` pulls in statsmodels/scikit-learn/scipy
# (and, for GAMM, bambi) - optional dependencies of the `ecoscope` package
# itself. Imported lazily inside the functions below, not at module level,
# so an environment missing them only breaks trend-analysis tasks rather
# than every task in `ecoscope.platform.tasks` (this module is imported
# eagerly by `ecoscope.platform.tasks.analysis.__init__`).


class LinearTrendModel(BaseModel):
    """Ordinary least squares. No smoothing - a single straight-line trend."""

    model_config = ConfigDict(title="Linear Regression")
    model: Annotated[Literal["linear"], Field(default="linear", title="Model")] = "linear"
    add_intercept: Annotated[bool, AdvancedField(True, title="Add Intercept")] = True


class GlmTrendModel(BaseModel):
    """Generalized linear model. Baseline for comparison against the smoothed
    models below - not recommended as the final trend line."""

    model_config = ConfigDict(title="Generalized Linear Model (GLM)")
    model: Annotated[Literal["glm"], Field(default="glm", title="Model")] = "glm"
    family: Annotated[
        Literal["gaussian", "poisson", "binomial", "gamma"],
        AdvancedField("gaussian", title="Distribution Family"),
    ] = "gaussian"
    add_intercept: Annotated[bool, AdvancedField(True, title="Add Intercept")] = True


class GamTrendModel(BaseModel):
    """Generalized additive model via B-splines. Smooths the trend rather
    than fitting a straight line; the recommended default for most trend
    charts."""

    model_config = ConfigDict(title="Generalized Additive Model (GAM)")
    model: Annotated[Literal["gam"], Field(default="gam", title="Model")] = "gam"
    family: Annotated[
        Literal["gaussian", "poisson", "binomial"],
        AdvancedField("gaussian", title="Distribution Family"),
    ] = "gaussian"
    alpha: Annotated[
        float | None,
        AdvancedField(
            None,
            title="Smoothing Parameter (Alpha)",
            description="Fixed smoothing strength. Leave empty to select automatically via cross-validation.",
        ),
    ] = None
    metric: Annotated[
        Literal["aic", "bic", "euclidean", "mse", "r_squared"],
        AdvancedField("aic", title="Alpha Selection Metric", description="Used only when Alpha is left empty."),
    ] = "aic"
    degree_of_freedom: Annotated[int, AdvancedField(20, title="Spline Degrees of Freedom")] = 20
    degree: Annotated[int, AdvancedField(3, title="Spline Degree")] = 3
    lower_bound: Annotated[float | None, AdvancedField(None, title="Lower Knot Bound")] = None
    upper_bound: Annotated[float | None, AdvancedField(None, title="Upper Knot Bound")] = None


class GammTrendModel(BaseModel):
    """Bayesian generalized additive mixed model with per-site random
    effects. Accounts for repeated measurements at the same site, at the
    cost of MCMC sampling - both fitting and (since a fitted estimator can't
    cross a task boundary) every downstream prediction refit this model from
    scratch, making it substantially slower than the other three."""

    model_config = ConfigDict(title="Generalized Additive Mixed Model (GAMM)")
    model: Annotated[Literal["gamm"], Field(default="gamm", title="Model")] = "gamm"
    site_id_column: Annotated[
        str,
        Field(
            title="Site/Group Column",
            description="Column identifying which site or group each row belongs to, fitted as a random effect.",
        ),
    ]
    family: Annotated[
        Literal["gaussian", "poisson", "gamma", "bernoulli"],
        AdvancedField("gaussian", title="Distribution Family"),
    ] = "gaussian"
    degree_of_freedom: Annotated[int, AdvancedField(10, title="Spline Degrees of Freedom")] = 10
    inference_method: Annotated[Literal["mcmc", "laplace"], AdvancedField("mcmc", title="Inference Method")] = "mcmc"
    draws: Annotated[int, AdvancedField(500, title="Posterior Draws")] = 500
    tune: Annotated[int | None, AdvancedField(None, title="MCMC Tuning Steps")] = None
    chains: Annotated[int, AdvancedField(2, title="MCMC Chains")] = 2
    random_seed: Annotated[int | None, AdvancedField(None, title="Random Seed")] = None


TrendModel: TypeAlias = Annotated[
    LinearTrendModel | GlmTrendModel | GamTrendModel | GammTrendModel,
    Field(discriminator="model"),
]

_TREND_MODEL_TYPES: dict[str, type[BaseModel]] = {
    "linear": LinearTrendModel,
    "glm": GlmTrendModel,
    "gam": GamTrendModel,
    "gamm": GammTrendModel,
}

_DEFAULT_TREND_MODEL: dict = {"model": "gam"}


def _build_regressor(model: TrendModel):
    from ecoscope.analysis.trend_analysis import (  # type: ignore[import-untyped]
        GAMMRegressor,
        GAMRegressor,
        GLMRegressor,
        LinearRegressionRegressor,
    )

    if isinstance(model, LinearTrendModel):
        return LinearRegressionRegressor(add_intercept=model.add_intercept)
    if isinstance(model, GlmTrendModel):
        return GLMRegressor(family=model.family, add_intercept=model.add_intercept)
    if isinstance(model, GamTrendModel):
        return GAMRegressor(
            alpha=model.alpha,
            degree_of_freedom=model.degree_of_freedom,
            degree=model.degree,
            family=model.family,
        )
    return GAMMRegressor(
        degree_of_freedom=model.degree_of_freedom,
        inference_method=model.inference_method,
        draws=model.draws,
        tune=model.tune,
        chains=model.chains,
        family=model.family,
        random_seed=model.random_seed,
    )


def _fit_regressor(model: TrendModel, X: np.ndarray, y: np.ndarray, site_ids: np.ndarray | None):
    regressor = _build_regressor(model)
    if isinstance(model, GamTrendModel):
        return regressor.fit(
            X,
            y,
            lower_bound=model.lower_bound,
            upper_bound=model.upper_bound,
            metric=model.metric,
        )
    if isinstance(model, GammTrendModel):
        return regressor.fit(X, y, site_ids)
    return regressor.fit(X, y)


def _prepare_xy(dataframe: pd.DataFrame, time_column: str, value_column: str) -> tuple[np.ndarray, np.ndarray]:
    X = dataframe[time_column].to_numpy()
    y = dataframe[value_column].to_numpy()
    if pd.api.types.is_datetime64_any_dtype(dataframe[time_column]):
        X = pd.to_numeric(dataframe[time_column]).to_numpy()
    return X, y


@register()
def set_trend_model(
    model: Annotated[TrendModel, Field(title="Trend Model", default=_DEFAULT_TREND_MODEL)],
) -> TrendModel:
    return model


@register()
def fit_trend_model(
    dataframe: Annotated[AnyDataFrame, Field(description="DataFrame containing time series data")],
    model: TrendModel,
    time_column: Annotated[str, Field(description="Column name containing time/date values")] = "time",
    value_column: Annotated[str, Field(description="Column name containing values to analyze")] = "value",
) -> dict:
    """Fit the selected trend model to `dataframe` and return its (re-fittable) parameters.

    The fitted estimator itself can't cross a task boundary, so this stores the
    model config plus the raw X/y arrays; `predict_trend_model` refits from them.
    """
    X, y = _prepare_xy(dataframe, time_column, value_column)

    site_ids = None
    if isinstance(model, GammTrendModel):
        if model.site_id_column not in dataframe.columns:
            raise ValueError(f"site_id_column {model.site_id_column!r} not found in dataframe")
        site_ids = dataframe[model.site_id_column].to_numpy()

    regressor = _fit_regressor(model, X, y, site_ids)

    metrics: dict[str, float] = {
        "r_squared": regressor.r_squared(X, y),
        "mse": regressor.mse(X, y),
    }
    try:
        metrics["aic"] = regressor.aic()
        metrics["bic"] = regressor.bic()
    except NotImplementedError:
        pass  # GAMM is Bayesian; aic/bic don't apply (see GAMMRegressor.aic/bic).

    return {
        "model": model.model_dump(),
        "X": X.tolist(),
        "y": y.tolist(),
        "site_ids": site_ids.tolist() if site_ids is not None else None,
        "metrics": metrics,
    }


@register()
def predict_trend_model(
    model_params: Annotated[dict, Field(description="Model parameters from fit_trend_model")],
    time_values: Annotated[
        list[float] | None,
        Field(default=None, description="Time values for prediction. If None, uses original training times."),
    ] = None,
    include_ci: Annotated[bool, Field(default=True, description="Include confidence intervals")] = True,
) -> AnyDataFrame:
    """Refit the selected trend model from stored parameters and predict - a
    fitted estimator can't cross a task boundary (see `fit_trend_model`)."""
    model = cast(TrendModel, _TREND_MODEL_TYPES[model_params["model"]["model"]](**model_params["model"]))
    X = np.asarray(model_params["X"])
    y = np.asarray(model_params["y"])
    site_ids = np.asarray(model_params["site_ids"]) if model_params.get("site_ids") is not None else None

    regressor = _fit_regressor(model, X, y, site_ids)

    predict_at = np.asarray(time_values) if time_values is not None else X
    observed = y if time_values is None else np.full(len(predict_at), np.nan)

    if include_ci:
        mean, ci_lower, ci_upper = regressor.predict_with_ci(predict_at)
        result = pd.DataFrame(
            {"y": observed, "time": predict_at, "predicted": mean, "ci_lower": ci_lower, "ci_upper": ci_upper}
        )
    else:
        result = pd.DataFrame({"y": observed, "time": predict_at, "predicted": regressor.predict(predict_at)})

    return cast(AnyDataFrame, result)
