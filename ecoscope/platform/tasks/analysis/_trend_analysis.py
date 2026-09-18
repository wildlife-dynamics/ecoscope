from typing import Annotated, Any, Literal, TypeAlias, cast

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field
from pydantic.functional_validators import BeforeValidator
from pydantic.json_schema import SkipJsonSchema
from wt_registry import register

from ecoscope.platform.annotations import AdvancedField, AnyDataFrame


def _advanced_titled_enum(*options: tuple[str, str]):
    """Field-level json_schema_extra: mark advanced and swap a Literal's bare
    enum for labeled options - same technique as `_unit.labeled_units`, but
    `AdvancedField` can't take a callable (it merges json_schema_extra via
    `|`, which only works for dicts), so this sets `ecoscope:advanced`
    itself instead of going through `AdvancedField`.
    """

    def apply(schema: dict) -> None:
        schema.pop("enum", None)
        schema["oneOf"] = [{"const": value, "title": title} for value, title in options]
        schema["ecoscope:advanced"] = True

    return apply


def _nullable_advanced(json_type: str):
    """Field-level json_schema_extra: mark advanced and rewrite pydantic's
    `anyOf: [{type: X}, {type: null}]` into the flat `type: [X, "null"]`
    array-of-types shorthand - equivalent JSON Schema, but the form this
    pipeline's RJSF/AJV setup actually needs to treat a number field as
    genuinely optional (confirmed via LocalFileSpatialFeatures.layer's
    existing working `type: ["string", "null"]` override; pydantic's own
    `anyOf` form does not work here). `AdvancedField` can't take a callable
    json_schema_extra (see `_advanced_titled_enum`), so this sets
    `ecoscope:advanced` itself instead of going through `AdvancedField`.
    """

    def apply(schema: dict) -> None:
        schema.pop("anyOf", None)
        schema["type"] = [json_type, "null"]
        schema["ecoscope:advanced"] = True

    return apply


def _empty_string_to_none(number_type: type):
    """BeforeValidator factory for a field typed `X | None` (see e.g.
    GamSmoothingSettings.alpha). Runs before pydantic's own type
    validation, normalizing an empty or numeric string into a real
    None/number - a defensive backstop in case a caller submits "" or a
    numeric string instead of a real number.
    """

    def validate(value: Any) -> Any:
        if value is None or value == "":
            return None
        if isinstance(value, str):
            return number_type(value)
        return value

    return validate


# `ecoscope.analysis.trend_analysis` pulls in statsmodels/scikit-learn/scipy
# (and, for GAMM, bambi) - optional dependencies of the `ecoscope` package
# itself. Imported lazily inside the functions below, not at module level,
# so an environment missing them only breaks trend-analysis tasks rather
# than every task in `ecoscope.platform.tasks` (this module is imported
# eagerly by `ecoscope.platform.tasks.analysis.__init__`).

# Every non-discriminator field below lives inside its own small settings
# model rather than directly on the variant class. `wt_registry`'s schema
# generator drops the pydantic `discriminator` keyword for function
# parameters (a workaround gap for https://github.com/pydantic/pydantic/issues/9404),
# so this union compiles down to a bare `anyOf` - the same shape as
# home-range's `HomeRangeMethodArgs`, and subject to the same RJSF
# field-blanking bug documented in that repo's
# docs/rjsf-anyof-switch-blanking-bug.md: a bare scalar field inside an
# anyOf branch is permanently blanked (not reset to its default) the first
# time a user switches the dropdown away from that branch and back. Giving
# every field real nested `properties` of its own (wrapping it) makes it
# self-heal instead. No docstring on any wrapper class deliberately -
# pydantic renders a model's docstring as its schema "description", which
# would leak this implementation note into the rendered form.


class AddInterceptSettings(BaseModel):
    model_config = ConfigDict(extra="ignore")
    add_intercept: Annotated[
        bool,
        AdvancedField(
            True,
            title="Add Intercept",
            description="Whether to fit a y-intercept term. Disable to force the line through the origin.",
        ),
    ] = True


class GlmFamilySettings(BaseModel):
    model_config = ConfigDict(extra="ignore")
    family: Annotated[
        Literal["gaussian", "poisson", "binomial", "gamma"],
        Field(
            default="gaussian",
            title="Distribution Family",
            description="Distribution assumed for the response variable.",
            json_schema_extra=_advanced_titled_enum(
                ("gaussian", "Gaussian"),
                ("poisson", "Poisson"),
                ("binomial", "Binomial"),
                ("gamma", "Gamma"),
            ),
        ),
    ] = "gaussian"


class GamFamilySettings(BaseModel):
    model_config = ConfigDict(extra="ignore")
    family: Annotated[
        Literal["gaussian", "poisson", "binomial"],
        Field(
            default="gaussian",
            title="Distribution Family",
            description="Distribution assumed for the response variable.",
            json_schema_extra=_advanced_titled_enum(
                ("gaussian", "Gaussian"),
                ("poisson", "Poisson"),
                ("binomial", "Binomial"),
            ),
        ),
    ] = "gaussian"


class GamSmoothingSettings(BaseModel):
    model_config = ConfigDict(extra="ignore")
    alpha: Annotated[
        float | None,
        BeforeValidator(_empty_string_to_none(float)),
        Field(
            default=None,
            title="Smoothing Parameter (Alpha)",
            description="Fixed smoothing strength. Leave empty to select automatically via cross-validation.",
            json_schema_extra=_nullable_advanced("number"),
        ),
    ] = None
    metric: Annotated[
        Literal["aic", "bic", "euclidean", "mse", "r_squared"],
        Field(
            default="aic",
            title="Alpha Selection Metric",
            description="Used only when Alpha is left empty.",
            json_schema_extra=_advanced_titled_enum(
                ("aic", "AIC"),
                ("bic", "BIC"),
                ("euclidean", "Euclidean Distance"),
                ("mse", "MSE"),
                ("r_squared", "R-Squared"),
            ),
        ),
    ] = "aic"


class GamSplineSettings(BaseModel):
    model_config = ConfigDict(extra="ignore")
    degree_of_freedom: Annotated[
        int,
        AdvancedField(
            6,
            title="Spline Degrees of Freedom",
            description="Number of basis functions for the spline. Higher values allow more "
            "flexible curves but risk overfitting.",
        ),
    ] = 6
    degree: Annotated[
        int,
        AdvancedField(3, title="Spline Degree", description="Polynomial degree of each spline segment (3 = cubic)."),
    ] = 3


class GamBoundsSettings(BaseModel):
    model_config = ConfigDict(extra="ignore")
    # Excluded rather than exposed: unlike alpha/tune/random_seed, these are
    # edge-case tuning knobs almost nobody needs to override, and "infer
    # from the data range" (the None default) is what GAMRegressor.fit
    # itself does whenever these aren't given - not worth exposing a
    # nullable-number field in the form for. Always None; never shown.
    lower_bound: Annotated[
        float | SkipJsonSchema[None],
        Field(
            default=None,
            title="Lower Knot Bound",
            description="Lower bound for spline knot placement. Leave empty to infer from the data range.",
            exclude=True,
        ),
    ] = None
    upper_bound: Annotated[
        float | SkipJsonSchema[None],
        Field(
            default=None,
            title="Upper Knot Bound",
            description="Upper bound for spline knot placement. Leave empty to infer from the data range.",
            exclude=True,
        ),
    ] = None


class GammFamilySettings(BaseModel):
    model_config = ConfigDict(extra="ignore")
    family: Annotated[
        Literal["gaussian", "poisson", "gamma", "bernoulli"],
        Field(
            default="gaussian",
            title="Distribution Family",
            description="Distribution assumed for the response variable.",
            json_schema_extra=_advanced_titled_enum(
                ("gaussian", "Gaussian"),
                ("poisson", "Poisson"),
                ("gamma", "Gamma"),
                ("bernoulli", "Bernoulli"),
            ),
        ),
    ] = "gaussian"


class GammSplineSettings(BaseModel):
    model_config = ConfigDict(extra="ignore")
    degree_of_freedom: Annotated[
        int,
        AdvancedField(
            10,
            title="Spline Degrees of Freedom",
            description="Number of basis functions for the spline. Higher values allow more "
            "flexible curves but risk overfitting.",
        ),
    ] = 10


class GammMcmcSettings(BaseModel):
    model_config = ConfigDict(extra="ignore")
    inference_method: Annotated[
        Literal["mcmc", "laplace"],
        Field(
            default="mcmc",
            title="Inference Method",
            description="MCMC is the reliable default for models with random effects; "
            "Laplace is faster when it converges but may fail for some model specifications.",
            json_schema_extra=_advanced_titled_enum(("mcmc", "MCMC"), ("laplace", "Laplace")),
        ),
    ] = "mcmc"
    draws: Annotated[
        int,
        AdvancedField(500, title="Posterior Draws", description="Number of posterior samples to draw per chain."),
    ] = 500
    tune: Annotated[
        int | None,
        BeforeValidator(_empty_string_to_none(int)),
        Field(
            default=None,
            title="MCMC Tuning Steps",
            description="Number of MCMC tuning steps. Leave empty to default to the same value as Posterior Draws.",
            json_schema_extra=_nullable_advanced("integer"),
        ),
    ] = None
    chains: Annotated[
        int,
        AdvancedField(2, title="MCMC Chains", description="Number of independent MCMC chains to run."),
    ] = 2
    random_seed: Annotated[
        int | None,
        BeforeValidator(_empty_string_to_none(int)),
        Field(
            default=None,
            title="Random Seed",
            description="Seed for the MCMC sampler. Leave empty for a non-deterministic fit.",
            json_schema_extra=_nullable_advanced("integer"),
        ),
    ] = None


class LinearTrendModel(BaseModel):
    """Ordinary least squares. No smoothing - a single straight-line trend."""

    model_config = ConfigDict(title="Linear Regression")
    model: Annotated[Literal["linear"], Field(default="linear", title="Model")] = "linear"
    intercept_settings: Annotated[
        AddInterceptSettings,
        AdvancedField(AddInterceptSettings(), title="Add Intercept"),
    ] = AddInterceptSettings()

    def get_params(self) -> dict:
        return {"add_intercept": self.intercept_settings.add_intercept}


class GlmTrendModel(BaseModel):
    """Generalized linear model. Baseline for comparison against the smoothed
    models below - not recommended as the final trend line."""

    model_config = ConfigDict(title="Generalized Linear Model (GLM)")
    model: Annotated[Literal["glm"], Field(default="glm", title="Model")] = "glm"
    family_settings: Annotated[
        GlmFamilySettings,
        AdvancedField(GlmFamilySettings(), title="Distribution Family"),
    ] = GlmFamilySettings()
    intercept_settings: Annotated[
        AddInterceptSettings,
        AdvancedField(AddInterceptSettings(), title="Add Intercept"),
    ] = AddInterceptSettings()

    def get_params(self) -> dict:
        return {
            "family": self.family_settings.family,
            "add_intercept": self.intercept_settings.add_intercept,
        }


class GamTrendModel(BaseModel):
    """Generalized additive model via B-splines. Smooths the trend rather
    than fitting a straight line; the recommended default for most trend
    charts."""

    model_config = ConfigDict(title="Generalized Additive Model (GAM)")
    model: Annotated[Literal["gam"], Field(default="gam", title="Model")] = "gam"
    family_settings: Annotated[
        GamFamilySettings,
        AdvancedField(GamFamilySettings(), title="Distribution Family"),
    ] = GamFamilySettings()
    smoothing_settings: Annotated[
        GamSmoothingSettings,
        AdvancedField(GamSmoothingSettings(), title="Smoothing"),
    ] = GamSmoothingSettings()
    spline_settings: Annotated[
        GamSplineSettings,
        AdvancedField(GamSplineSettings(), title="Spline Shape"),
    ] = GamSplineSettings()
    # Both of GamBoundsSettings' own fields are excluded (see its docstring) -
    # nothing left to show, so the wrapper itself is excluded too, rather
    # than rendering an empty "Knot Bounds" section.
    bounds_settings: Annotated[
        GamBoundsSettings,
        AdvancedField(GamBoundsSettings(), title="Knot Bounds", exclude=True),
    ] = GamBoundsSettings()

    def get_params(self) -> dict:
        return {
            "family": self.family_settings.family,
            "alpha": self.smoothing_settings.alpha,
            "metric": self.smoothing_settings.metric,
            "degree_of_freedom": self.spline_settings.degree_of_freedom,
            "degree": self.spline_settings.degree,
            "lower_bound": self.bounds_settings.lower_bound,
            "upper_bound": self.bounds_settings.upper_bound,
        }


class GammTrendModel(BaseModel):
    """Bayesian generalized additive mixed model with per-site random
    effects. A workflow fits this once across every site's combined data
    (site_ids derived automatically from a "name" column if present - see
    fit_trend_model), then predicts each site's own group-specific curve
    from that shared fit (see predict_trend_model's `dataframe` parameter) -
    unlike the other three models, which each fit and predict independently
    per site. Mainly useful here for its Bayesian credible intervals rather
    than frequentist confidence intervals, at the added cost of MCMC
    sampling - both fitting and (since a fitted estimator can't cross a task
    boundary) every downstream prediction refit this model from scratch,
    making it substantially slower than the other three."""

    model_config = ConfigDict(title="Generalized Additive Mixed Model (GAMM)")
    model: Annotated[Literal["gamm"], Field(default="gamm", title="Model")] = "gamm"
    family_settings: Annotated[
        GammFamilySettings,
        AdvancedField(GammFamilySettings(), title="Distribution Family"),
    ] = GammFamilySettings()
    spline_settings: Annotated[
        GammSplineSettings,
        AdvancedField(GammSplineSettings(), title="Spline Shape"),
    ] = GammSplineSettings()
    mcmc_settings: Annotated[
        GammMcmcSettings,
        AdvancedField(GammMcmcSettings(), title="MCMC Sampling"),
    ] = GammMcmcSettings()

    def get_params(self) -> dict:
        return {
            "family": self.family_settings.family,
            "degree_of_freedom": self.spline_settings.degree_of_freedom,
            "inference_method": self.mcmc_settings.inference_method,
            "draws": self.mcmc_settings.draws,
            "tune": self.mcmc_settings.tune,
            "chains": self.mcmc_settings.chains,
            "random_seed": self.mcmc_settings.random_seed,
        }


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

# RJSF's MultiSchemaField only expands a partial default (e.g. {"model": "gam"})
# into its full set of nested field defaults inside onOptionChange - i.e. only
# when the user actually interacts with the dropdown. On first form load, the
# literal default below is used as-is, with no such expansion. So this must be
# the FULLY expanded dict (every nested settings field included), or every
# advanced field under the default-selected model renders empty until the user
# touches the dropdown at least once.
_DEFAULT_TREND_MODEL: dict = GamTrendModel().model_dump()


def _build_regressor(model: TrendModel):
    from ecoscope.analysis.trend_analysis import (  # type: ignore[import-untyped]
        GAMMRegressor,
        GAMRegressor,
        GLMRegressor,
        LinearRegressionRegressor,
    )

    if isinstance(model, LinearTrendModel):
        return LinearRegressionRegressor(**model.get_params())
    if isinstance(model, GlmTrendModel):
        return GLMRegressor(**model.get_params())
    if isinstance(model, GamTrendModel):
        params = model.get_params()
        return GAMRegressor(
            alpha=params["alpha"],
            degree_of_freedom=params["degree_of_freedom"],
            degree=params["degree"],
            family=params["family"],
        )
    params = model.get_params()
    return GAMMRegressor(
        degree_of_freedom=params["degree_of_freedom"],
        inference_method=params["inference_method"],
        draws=params["draws"],
        tune=params["tune"],
        chains=params["chains"],
        family=params["family"],
        random_seed=params["random_seed"],
    )


def _fit_regressor(model: TrendModel, X: np.ndarray, y: np.ndarray, site_ids: np.ndarray | None):
    regressor = _build_regressor(model)
    if isinstance(model, GamTrendModel):
        params = model.get_params()
        return regressor.fit(
            X,
            y,
            lower_bound=params["lower_bound"],
            upper_bound=params["upper_bound"],
            metric=params["metric"],
        )
    if isinstance(model, GammTrendModel):
        return regressor.fit(X, y, site_ids)
    return regressor.fit(X, y)


def _prepare_xy(dataframe: pd.DataFrame, time_column: str, value_column: str) -> tuple[np.ndarray, np.ndarray]:
    X = dataframe[time_column].to_numpy()
    y = dataframe[value_column].to_numpy()
    if pd.api.types.is_datetime64_any_dtype(dataframe[time_column]):
        # Days (not raw nanoseconds) since the epoch: nanosecond timestamps
        # (~1e18) make the OLS/GLM design matrix numerically singular once an
        # intercept is added (condition number ~1e19, beyond float64
        # precision) - days (~1e4-1e5) keep it well-conditioned while still
        # being a fixed, deterministic transform (no per-dataset state to
        # keep in sync between fit and predict). Converts via the Series
        # itself (not `.to_numpy()`) so a timezone-aware column converts
        # cleanly to UTC nanoseconds instead of numpy warning about dropped
        # timezone info.
        X = dataframe[time_column].astype("int64").to_numpy() / 1e9 / 86400.0
    return X, y


@register()
def set_trend_model(
    model: Annotated[TrendModel, Field(title="Trend Model", default=_DEFAULT_TREND_MODEL)],
) -> TrendModel:
    return model


@register()
def is_gamm_trend_model(*args: Any) -> bool:
    """skipif condition: True if any arg is a GammTrendModel.

    GAMM fits once across every site's combined data (see GammTrendModel's
    docstring), unlike the other 3 models which each fit independently per
    site - so a workflow wires two separate fit branches (per-site, and one
    combined across sites) and uses this (and its inverse) to run only the
    one that matches the selection.
    """
    return any(isinstance(a, GammTrendModel) for a in args)


@register()
def is_not_gamm_trend_model(*args: Any) -> bool:
    return not is_gamm_trend_model(*args)


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
        # GAMMRegressor requires site_ids. Use the "name" column if
        # extract_forest_cover_trends propagated one (see its own
        # docstring) - real cross-site variance when `dataframe` spans
        # multiple sites (a combined fit), a single repeated value
        # otherwise - or a constant placeholder if "name" isn't present at
        # all.
        site_ids = dataframe["name"].to_numpy() if "name" in dataframe.columns else np.zeros(len(y))

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
        # Whether `time_column` was a datetime column, so predict_trend_model
        # knows to convert its "time" output back from the days-since-epoch
        # _prepare_xy fits on into a real timestamp (see _prepare_xy).
        "time_is_datetime": bool(pd.api.types.is_datetime64_any_dtype(dataframe[time_column])),
    }


@register()
def predict_trend_model(
    model_params: Annotated[dict, Field(description="Model parameters from fit_trend_model")],
    dataframe: Annotated[
        # None must come first: with AnyDataFrame first, pydantic's union
        # validation coerces an explicit `dataframe=None` (always passed
        # explicitly once this parameter is used in any mapvalues() call -
        # see wt_task's _get_defaults) into an EMPTY DataFrame via pandera's
        # DataFrame[Schema] validator, rather than preserving None.
        SkipJsonSchema[None] | AnyDataFrame,
        Field(
            default=None,
            description="Predict at this dataframe's own time/value columns instead of the original "
            "training data - e.g. a single site's own rows, to get that site's group-specific "
            "prediction out of a GAMM model fit combined across multiple sites (matched via this "
            "dataframe's own 'name' column, if present, against a value in that fit's site_ids). "
            "Takes precedence over time_values if both are given.",
            exclude=True,
        ),
    ] = None,
    time_column: Annotated[
        str, Field(description="Column name containing time/date values, if `dataframe` is given")
    ] = "time",
    value_column: Annotated[
        str, Field(description="Column name containing observed values, if `dataframe` is given")
    ] = "value",
    time_values: Annotated[
        list[float] | SkipJsonSchema[None],
        Field(
            default=None,
            description="Time values for prediction, if `dataframe` is not given. "
            "If neither is given, uses original training times.",
        ),
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

    predict_site_id = None
    if dataframe is not None:
        predict_at, observed = _prepare_xy(dataframe, time_column, value_column)
        if "name" in dataframe.columns:
            predict_site_id = dataframe["name"].iloc[0]
    elif time_values is not None:
        predict_at = np.asarray(time_values)
        observed = np.full(len(predict_at), np.nan)
    else:
        predict_at = X
        observed = y

    predict_kwargs: dict = {}
    if isinstance(model, GammTrendModel) and predict_site_id is not None:
        predict_kwargs["site_ids"] = np.full(len(predict_at), predict_site_id)

    if include_ci:
        mean, ci_lower, ci_upper = regressor.predict_with_ci(predict_at, **predict_kwargs)
        result = pd.DataFrame(
            {"y": observed, "time": predict_at, "predicted": mean, "ci_lower": ci_lower, "ci_upper": ci_upper}
        )
    else:
        result = pd.DataFrame(
            {"y": observed, "time": predict_at, "predicted": regressor.predict(predict_at, **predict_kwargs)}
        )

    if model_params.get("time_is_datetime"):
        # predict_at is in days-since-epoch (see _prepare_xy) - convert back
        # to a real timestamp so "time" isn't a bare number a caller has to
        # know the unit of (and previously, before _prepare_xy switched to
        # days for numerical stability, silently mis-happened to work only
        # because raw nanoseconds-since-epoch is pd.to_datetime's own
        # default unit).
        result["time"] = pd.to_datetime(result["time"], unit="D")

    return cast(AnyDataFrame, result)


@register()
def get_trend_model_fit_summary(
    model_params: Annotated[dict, Field(description="Model parameters from fit_trend_model")],
    model: Annotated[
        TrendModel,
        Field(
            description="The same model passed to fit_trend_model - not read from model_params directly "
            "(a plain dict) so that is_gamm_trend_model/is_not_gamm_trend_model skipif conditions, which "
            "type-check the real model object, work on this task too.",
            exclude=True,
        ),
    ],
) -> Annotated[
    AnyDataFrame,
    Field(
        description="Fit parameter summary (e.g. coefficient/std error/p-value/CI for Linear/GLM/GAM, or "
        "posterior mean/sd/hdi/r_hat for GAMM) - the direct output of the underlying statsmodels or "
        "pymc/bambi fit, one row per model parameter."
    ),
]:
    """Refit from `model_params` (a fitted estimator can't cross a task
    boundary; see `fit_trend_model`) and return its own fit parameter
    summary - e.g. to persist for inspection alongside the trend chart."""
    X = np.asarray(model_params["X"])
    y = np.asarray(model_params["y"])
    site_ids = np.asarray(model_params["site_ids"]) if model_params.get("site_ids") is not None else None

    regressor = _fit_regressor(model, X, y, site_ids)
    return cast(AnyDataFrame, regressor.summary())
