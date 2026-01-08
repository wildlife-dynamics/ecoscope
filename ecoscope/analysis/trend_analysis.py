"""
Todo: 
- Outputs a dataframe representing the GAMM for a unique dataset as a benchmark
- Extract forest cover as a task in Ecoscope. Workflow process
- 
"""



"""
Trend Analysis using Generalized Additive Models (GAMs).

This module provides tools for fitting GAMs to time series data,
particularly useful for analyzing environmental trends from remote sensing data.
"""

import logging
from typing import Literal, Optional, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd

try:
    import ee
except ImportError:
    ee = None

try:
    from scipy.spatial.distance import euclidean
    from sklearn.base import BaseEstimator, RegressorMixin
    from sklearn.model_selection import BaseCrossValidator, KFold, LeaveOneOut
    from statsmodels.gam.api import BSplines, GLMGam
    from statsmodels.genmod.families import Binomial, Gaussian, Poisson
except ModuleNotFoundError:
    raise ModuleNotFoundError(
        'Missing optional dependencies required by this module. '
        'Please run pip install ecoscope["analysis"]'
    )

try:
    import plotly.graph_objects as go
except ModuleNotFoundError:
    go = None

logger = logging.getLogger(__name__)


class GAMRegressor(BaseEstimator, RegressorMixin):
    """
    Generalized Additive Model (GAM) Regressor using B-Splines.

    A scikit-learn compatible wrapper around statsmodels GLMGam that provides
    a user-friendly interface for fitting GAMs to time series data.

    Parameters
    ----------
    alpha : float, default=0.1
        Smoothing parameter. Higher values result in smoother curves (more linear).
    df : int, default=20
        Degrees of freedom for the spline basis.
    degree : int, default=3
        Degree of the B-spline basis (cubic splines by default).
    family : {"gaussian", "poisson", "binomial"}, default="gaussian"
        Distribution family for the GLM.
    upper_bound : float, optional
        Upper bound for spline knots. If None, uses max(X).
    lower_bound : float, optional
        Lower bound for spline knots. If None, uses min(X).

    Examples
    --------
    >>> from ecoscope.analysis.trend_analysis import GAMRegressor
    >>> import numpy as np
    >>> X = np.array([2000, 2001, 2002, 2003, 2004]).reshape(-1, 1)
    >>> y = np.array([100, 95, 90, 85, 80])
    >>> gam = GAMRegressor(alpha=0.1).fit(X, y)
    >>> predictions = gam.predict(X)
    """

    def __init__(
        self,
        alpha: float = 0.1,
        df: int = 20,
        degree: int = 3,
        family: Literal["gaussian", "poisson", "binomial"] = "gaussian",
        upper_bound: Optional[float] = None,
        lower_bound: Optional[float] = None,
    ):
        self.alpha = alpha
        self.df = df
        self.degree = degree
        self.upper_bound = upper_bound
        self.lower_bound = lower_bound

        if family == "gaussian":
            self.family = Gaussian()
        elif family == "poisson":
            self.family = Poisson()
        elif family == "binomial":
            self.family = Binomial()
        else:
            raise ValueError(f"Unsupported family: {family}. Must be 'gaussian', 'poisson', or 'binomial'")

    def fit(self, X, y):
        """
        Fit the GAM model.

        Parameters
        ----------
        X : array-like of shape (n_samples, 1) or (n_samples,)
            Training data (typically time/date values).
        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        self : GAMRegressor
            Returns self for method chaining.
        """
        X = np.asarray(X)
        if X.ndim == 1:
            X = X[:, None]
        y = np.asarray(y).ravel()

        knot_kwds = {}
        if self.upper_bound is not None and self.lower_bound is not None:
            knot_kwds = [{"upper_bound": self.upper_bound, "lower_bound": self.lower_bound}]

        if knot_kwds:
            self._spline_ = BSplines(X, df=[self.df], degree=[self.degree], knot_kwds=knot_kwds)
        else:
            self._spline_ = BSplines(X, df=[self.df], degree=[self.degree])

        exog = np.ones((len(X), 1))

        self._res_ = GLMGam(y, exog=exog, smoother=self._spline_, alpha=self.alpha, family=self.family).fit()

        return self

    def predict(self, X):
        """
        Predict using the fitted model.

        Parameters
        ----------
        X : array-like of shape (n_samples, 1) or (n_samples,)
            Samples to predict.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted values.
        """
        X = np.asarray(X)
        if X.ndim == 1:
            X = X[:, None]

        exog = np.ones((len(X), 1))
        return self._res_.predict(exog=exog, exog_smooth=X)

    def aic(self) -> float:
        """Return Akaike Information Criterion."""
        return self._res_.aic

    def bic(self) -> float:
        """Return Bayesian Information Criterion."""
        return self._res_.bic_llf

    def predict_with_ci(self, X) -> Tuple[np.ndarray, np.ndarray, np.ndarray]: # custom confidence interval calculation
        """
        Predict with confidence intervals.

        Parameters
        ----------
        X : array-like of shape (n_samples, 1) or (n_samples,)
            Samples to predict.

        Returns
        -------
        mean : ndarray
            Predicted mean values.
        ci_lower : ndarray
            Lower bound of confidence interval.
        ci_upper : ndarray
            Upper bound of confidence interval.
        """
        X = np.asarray(X)
        if X.ndim == 1:
            X = X[:, None]

        exog = np.ones((len(X), 1))
        summary_frame = self._res_.get_prediction(exog=exog, exog_smooth=X).summary_frame()
        return (
            summary_frame["mean"].to_numpy(),
            summary_frame["mean_ci_lower"].to_numpy(),
            summary_frame["mean_ci_upper"].to_numpy(),
        )


def choose_cross_validator(X: np.ndarray) -> BaseCrossValidator:
    """
    Choose appropriate cross-validator based on sample size.

    Parameters
    ----------
    X : ndarray
        Input data.

    Returns
    -------
    BaseCrossValidator
        Cross-validation strategy.
    """
    if len(X) <= 10:
        return LeaveOneOut()
    else:
        return KFold(n_splits=5)


def optimize_gam_cv(
    X: np.ndarray,
    y: np.ndarray,
    alphas: np.ndarray,
    cross_validator: BaseCrossValidator,
    metric: Literal["aic", "bic", "euclidean"] = "aic",
    lower_bound: Optional[float] = None,
    upper_bound: Optional[float] = None,
) -> Tuple[float, GAMRegressor]:
    """
    Optimize GAM smoothing parameter using cross-validation.

    Parameters
    ----------
    X : ndarray
        Training data.
    y : ndarray
        Target values.
    alphas : ndarray
        Array of alpha values to search.
    cross_validator : BaseCrossValidator
        Cross-validation strategy.
    metric : {"aic", "bic", "euclidean"}, default="aic"
        Metric to optimize.
    lower_bound : float, optional
        Lower bound for spline knots.
    upper_bound : float, optional
        Upper bound for spline knots.

    Returns
    -------
    best_alpha : float
        Optimal alpha value.
    best_gam : GAMRegressor
        Fitted GAM with optimal alpha.
    """
    n_folds = cross_validator.get_n_splits(X)
    gridsearch_matrix = np.zeros((len(alphas), n_folds))

    for i, alpha in enumerate(alphas):
        for j, (train_index, test_index) in enumerate(cross_validator.split(X)):
            X_train = X[train_index]
            y_train = y[train_index]
            X_test = X[test_index]
            y_test = y[test_index]

            gam = GAMRegressor(alpha=alpha, lower_bound=lower_bound, upper_bound=upper_bound).fit(X_train, y_train)
            y_pred = gam.predict(X_test)

            if metric == "euclidean":
                gridsearch_matrix[i][j] = euclidean(y_test, y_pred)
            elif metric == "aic":
                gridsearch_matrix[i][j] = gam.aic()
            elif metric == "bic":
                gridsearch_matrix[i][j] = gam.bic()

    mean_score = np.mean(gridsearch_matrix, axis=1)
    best_alpha = alphas[np.argmin(mean_score)]
    best_gam = GAMRegressor(alpha=best_alpha, lower_bound=lower_bound, upper_bound=upper_bound).fit(X, y)

    return best_alpha, best_gam


def optimize_gam(
    X: np.ndarray,
    y: np.ndarray,
    cross_validator: Optional[BaseCrossValidator] = None,
    alphas: Optional[np.ndarray] = None,
    lower_bound: Optional[float] = None,
    upper_bound: Optional[float] = None,
    metric: Literal["aic", "bic", "euclidean"] = "aic",
) -> Tuple[float, GAMRegressor]:
    """
    Optimize GAM smoothing parameter with automatic defaults.

    Parameters
    ----------
    X : ndarray
        Training data.
    y : ndarray
        Target values.
    cross_validator : BaseCrossValidator, optional
        Cross-validation strategy. If None, chosen automatically.
    alphas : ndarray, optional
        Array of alpha values to search. Defaults to logspace(-4, 4, 100).
    lower_bound : float, optional
        Lower bound for spline knots. Defaults to min(X).
    upper_bound : float, optional
        Upper bound for spline knots. Defaults to max(X) + 5.
    metric : {"aic", "bic", "euclidean"}, default="aic"
        Metric to optimize.

    Returns
    -------
    best_alpha : float
        Optimal alpha value.
    best_gam : GAMRegressor
        Fitted GAM with optimal alpha.

    Examples
    --------
    >>> from ecoscope.analysis.trend_analysis import optimize_gam
    >>> import numpy as np
    >>> X = np.array([2000, 2001, 2002, 2003, 2004])
    >>> y = np.array([100, 95, 90, 85, 80])
    >>> alpha, gam = optimize_gam(X, y, metric="aic")
    """
    if lower_bound is None:
        lower_bound = float(np.min(X))
    if upper_bound is None:
        upper_bound = float(np.max(X)) + 5
    if alphas is None:
        alphas = np.logspace(-4, 4, 100)
    if cross_validator is None:
        cross_validator = choose_cross_validator(X)

    best_alpha, best_gam = optimize_gam_cv(
        X, y, alphas=alphas, cross_validator=cross_validator, metric=metric, lower_bound=lower_bound, upper_bound=upper_bound
    )

    return best_alpha, best_gam


def get_forest_cover_trends(
    aoi: gpd.GeoDataFrame,
    tree_cover_threshold: float = 60.0,
    scale: int = 30,
    max_pixels: int = 1e9,
) -> pd.DataFrame:
    """
    Extract forest cover trends from Google Earth Engine dataset.

    Parameters
    ----------
    aoi : gpd.GeoDataFrame
        Area of interest geometry (must have CRS set).
    tree_cover_threshold : float, default=60.0
        Minimum tree cover percentage to consider as forest (0-100).
    scale : int, default=30
        Pixel scale in meters for reduction.
    max_pixels : int, default=1e9
        Maximum pixels for reduction.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - year: Year of observation
        - loss_area: Forest loss area in acres for that year
        - cumsum_loss_area: Cumulative loss area in acres
        - survival_area: Remaining forest area in acres

    Examples
    --------
    >>> import geopandas as gpd
    >>> from ecoscope.analysis.trend_analysis import get_forest_cover_trends
    >>> aoi = gpd.GeoDataFrame(geometry=[...], crs=4326)
    >>> trends = get_forest_cover_trends(aoi)
    """
    if ee is None:
        raise ImportError("Google Earth Engine (earthengine-api) is required for this function.")

    if aoi.crs is None:
        logger.warning("AOI CRS not set. Assuming WGS84.")
        aoi = aoi.set_crs(4326)

    feat_coll = ee.FeatureCollection(aoi.__geo_interface__)
    gfc = ee.Image("UMD/hansen/global_forest_change_2023_v1_11")

    # Calculate forested area in 2000
    treecover2000 = gfc.select(["treecover2000"])
    treecover2000_mask = treecover2000.gte(tree_cover_threshold)
    treecover2000 = treecover2000.unmask().updateMask(treecover2000_mask)
    treecover2000 = treecover2000.And(treecover2000)  # Convert pixel values to 1's
    treecover2000_area_img = treecover2000.multiply(ee.Image.pixelArea())
    treecover2000_area = treecover2000_area_img.reduceRegion(
        reducer=ee.Reducer.sum(), geometry=feat_coll, scale=scale, maxPixels=max_pixels
    )

    forested_area = treecover2000_area.getInfo()["treecover2000"]
    forested_area = forested_area * 0.000247105  # Convert sq.meters to acres

    # Calculate forest loss by year
    loss_img = gfc.select(["loss"])
    loss_img = loss_img.updateMask(treecover2000_mask)
    loss_area_img = loss_img.multiply(ee.Image.pixelArea())
    loss_year = gfc.select(["lossyear"])

    loss_by_year = (
        loss_area_img.addBands(loss_year)
        .reduceRegion(reducer=ee.Reducer.sum().group(groupField=1), geometry=feat_coll, scale=scale, maxPixels=max_pixels)
    )

    forest_survival = pd.DataFrame([x for x in loss_by_year.getInfo()["groups"]])
    forest_survival.rename(columns={"group": "year", "sum": "loss_area"}, inplace=True)
    forest_survival["year"] = forest_survival["year"] + 2000
    forest_survival["loss_area"] = forest_survival["loss_area"] * 0.000247105  # Convert sq.meters to acres
    forest_survival["cumsum_loss_area"] = forest_survival["loss_area"].cumsum()
    forest_survival["survival_area"] = forested_area - forest_survival["cumsum_loss_area"]

    return forest_survival


def plot_trend(
    x: np.ndarray,
    y_orig: np.ndarray,
    y_mean: np.ndarray,
    y_lower: np.ndarray,
    y_upper: np.ndarray,
    plot_title: str = "Trend",
    xlabel: str = "Year",
    ylabel: str = "Value",
) -> "go.Figure":
    """
    Create a Plotly figure showing trend with confidence intervals.

    Parameters
    ----------
    x : ndarray
        X-axis values (e.g., years).
    y_orig : ndarray
        Original observed values.
    y_mean : ndarray
        Predicted mean values.
    y_lower : ndarray
        Lower confidence interval bound.
    y_upper : ndarray
        Upper confidence interval bound.
    plot_title : str, default="Trend"
        Plot title.
    xlabel : str, default="Year"
        X-axis label.
    ylabel : str, default="Value"
        Y-axis label.

    Returns
    -------
    go.Figure
        Plotly figure object.

    Examples
    --------
    >>> from ecoscope.analysis.trend_analysis import plot_trend
    >>> import numpy as np
    >>> x = np.array([2000, 2001, 2002])
    >>> y_orig = np.array([100, 95, 90])
    >>> y_mean = np.array([100, 95, 90])
    >>> y_lower = np.array([98, 93, 88])
    >>> y_upper = np.array([102, 97, 92])
    >>> fig = plot_trend(x, y_orig, y_mean, y_lower, y_upper, "Forest Cover")
    >>> fig.show()
    """
    if go is None:
        raise ImportError('Plotly is required for plotting. Install with: pip install ecoscope["plotting"]')

    fig = go.Figure(
        layout_yaxis_range=[float(np.min(y_lower)) * 0.95, float(np.max(y_upper)) * 1.2],
        layout_xaxis_range=[float(np.min(x)) - 2, float(np.max(x)) + 2],
    )

    fig.add_trace(go.Scatter(x=x, y=y_lower, line=dict(color="rgba(0,0,0,0)"), showlegend=False))

    fig.add_trace(
        go.Scatter(
            x=x,
            y=y_upper,
            fill="tonexty",
            fillcolor="rgba(0, 100, 200, 0.2)",
            line=dict(color="rgba(0,0,0,0)"),
            name="95% CI",
        )
    )

    fig.add_trace(go.Scatter(x=x, y=y_mean, line=dict(color="blue"), name="Trend"))

    fig.add_trace(
        go.Scatter(
            x=x,
            y=y_orig,
            mode="markers",
            marker=dict(color="rgba(1,50,32,1)", size=7),
            name="Raw Data",
        )
    )

    fig.update_layout(
        title=plot_title,
        xaxis_title=xlabel,
        yaxis_title=ylabel,
        legend_title="",
        autosize=False,
        width=500 * 1.73333,
        height=500,
    )

    return fig

