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

from typing import Literal, Optional, Tuple

import numpy as np
import pandas as pd

try:
    # type: ignore[import-untyped]
    from joblib import Parallel, delayed
    from scipy.spatial.distance import euclidean
    from sklearn.base import BaseEstimator, RegressorMixin
    from sklearn.model_selection import BaseCrossValidator, KFold, LeaveOneOut
    from statsmodels.gam.api import BSplines, GLMGam
    from statsmodels.genmod.families import Binomial, Gaussian, Poisson
except ModuleNotFoundError:
    raise ModuleNotFoundError(
        "Missing optional dependencies required by this module. " 'Please run pip install ecoscope["analysis"]'
    )

try:
    import plotly.graph_objects as go

    from ecoscope.plotting import draw_historic_timeseries
except ModuleNotFoundError:
    go = None
    draw_historic_timeseries = None


class GAMRegressor(BaseEstimator, RegressorMixin):
    """
    Generalized Additive Model (GAM) Regressor using B-Splines.

    A scikit-learn compatible wrapper around statsmodels GLMGam that provides
    a user-friendly interface for fitting GAMs to time series data.

    Parameters
    ----------
    alpha : float, default=0.1
        Smoothing parameter. Higher values result in smoother curves (more linear).
    degree_of_freedom : int, default=20
        Degrees of freedom for the spline basis.
    degree : int, default=3
        Degree of the B-spline basis (cubic splines by default).
    family : {"gaussian", "poisson", "binomial"}, default="gaussian"
        Distribution family for the GLM.

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
        degree_of_freedom: int = 20,
        degree: int = 3,
        family: Literal["gaussian", "poisson", "binomial"] = "gaussian",
    ):
        self.alpha = alpha
        self.degree_of_freedom = degree_of_freedom
        self.degree = degree

        if family == "gaussian":
            self.family = Gaussian()
        elif family == "poisson":
            self.family = Poisson()
        elif family == "binomial":
            self.family = Binomial()
        else:
            raise ValueError(f"Unsupported family: {family}. Must be 'gaussian', 'poisson', or 'binomial'")

    def fit(
        self,
        X,
        y,
        upper_bound: Optional[float] = None,
        lower_bound: Optional[float] = None,
    ):
        """
        Fit the GAM model.

        Parameters
        ----------
        X : array-like of shape (n_samples, 1) or (n_samples,)
            Training data (typically time/date values).
        y : array-like of shape (n_samples,)
            Target values.
        upper_bound : float, optional
            Upper bound for spline knots. If None, uses max(X).
        lower_bound : float, optional
            Lower bound for spline knots. If None, uses min(X).

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
        if upper_bound is not None and lower_bound is not None:
            knot_kwds = [{"upper_bound": upper_bound, "lower_bound": lower_bound}]

        if knot_kwds:
            self._spline_ = BSplines(X, df=[self.degree_of_freedom], degree=[self.degree], knot_kwds=knot_kwds)
        else:
            self._spline_ = BSplines(X, df=[self.degree_of_freedom], degree=[self.degree])

        exog = np.ones((len(X), 1))

        self._res_ = GLMGam(y, exog=exog, smoother=self._spline_, alpha=self.alpha, family=self.family).fit()

        return self

    def _check_is_fitted(self) -> None:
        """Check if the model has been fitted."""
        if not hasattr(self, "_res_"):
            raise ValueError("Model has not been fitted. Call fit() before using this method.")

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

        Raises
        ------
        ValueError
            If the model has not been fitted.
        """
        self._check_is_fitted()
        X = np.asarray(X)
        if X.ndim == 1:
            X = X[:, None]

        exog = np.ones((len(X), 1))
        return self._res_.predict(exog=exog, exog_smooth=X)

    def aic(self) -> float:
        """Return Akaike Information Criterion."""
        self._check_is_fitted()
        return self._res_.aic

    def bic(self) -> float:
        """Return Bayesian Information Criterion."""
        self._check_is_fitted()
        return self._res_.bic_llf

    def mse(self, X, y) -> float:
        """
        Return Mean Squared Error on given data.

        Parameters
        ----------
        X : array-like
            Input data.
        y : array-like
            True target values.

        Returns
        -------
        float
            Mean squared error.
        """
        self._check_is_fitted()
        y_pred = self.predict(X)
        y = np.asarray(y).ravel()
        return float(np.mean((y - y_pred) ** 2))

    def r_squared(self, X, y) -> float:
        """
        Return R-squared (coefficient of determination) on given data.

        Parameters
        ----------
        X : array-like
            Input data.
        y : array-like
            True target values.

        Returns
        -------
        float
            R-squared value. 1.0 indicates perfect fit, 0.0 indicates
            model performs same as predicting the mean.
        """
        self._check_is_fitted()
        y_pred = self.predict(X)
        y = np.asarray(y).ravel()
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        if ss_tot == 0:
            return 1.0 if ss_res == 0 else 0.0
        return float(1 - ss_res / ss_tot)

    def predict_with_ci(self, X) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:  # custom confidence interval calculation
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

        Raises
        ------
        ValueError
            If the model has not been fitted.
        """
        self._check_is_fitted()
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


def _fit_and_score_ic(alpha, X, y, metric, lower_bound, upper_bound, degree_of_freedom, degree, family):
    """Fit GAM and return alpha with its information criterion (aic, bic) score."""
    gam = GAMRegressor(
        alpha=alpha,
        degree_of_freedom=degree_of_freedom,
        degree=degree,
        family=family,
    ).fit(X, y, lower_bound=lower_bound, upper_bound=upper_bound)
    if metric == "aic":
        score = gam.aic()
    else:
        score = gam.bic()
    return alpha, score


def _fit_and_score_cv(
    alpha, fold_idx, train_index, test_index, X, y, metric, lower_bound, upper_bound, degree_of_freedom, degree, family
):
    """Fit GAM on fold of test/train data and return alpha, fold index, and score."""
    gam = GAMRegressor(
        alpha=alpha,
        degree_of_freedom=degree_of_freedom,
        degree=degree,
        family=family,
    ).fit(X[train_index], y[train_index], lower_bound=lower_bound, upper_bound=upper_bound)
    y_pred = gam.predict(X[test_index])
    y_test = y[test_index]

    if metric == "euclidean":
        score = euclidean(y_test, y_pred)
    elif metric == "mse":
        score = float(np.mean((y_test - y_pred) ** 2))
    elif metric == "r_squared":
        ss_res = np.sum((y_test - y_pred) ** 2)
        ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
        if ss_tot == 0:
            if ss_res == 0:
                score = 1.0
            else:
                score = 0.0
        else:
            score = 1 - ss_res / ss_tot
    return alpha, fold_idx, score


def optimize_gam_cv(
    X: np.ndarray,
    y: np.ndarray,
    alphas: np.ndarray,
    cross_validator: BaseCrossValidator,
    metric: Literal["aic", "bic", "euclidean", "mse", "r_squared"] = "aic",
    lower_bound: Optional[float] = None,
    upper_bound: Optional[float] = None,
    degree_of_freedom: int = 20,
    degree: int = 3,
    family: Literal["gaussian", "poisson", "binomial"] = "gaussian",
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
    metric : {"aic", "bic", "euclidean", "mse", "r_squared"}, default="aic"
        Metric to optimize. AIC/BIC are computed on full data, others use
        cross-validation. Note: r_squared is maximized, others are minimized.
    lower_bound : float, optional
        Lower bound for spline knots.
    upper_bound : float, optional
        Upper bound for spline knots.
    degree_of_freedom : int, default=20
        Degrees of freedom for the spline basis.
    degree : int, default=3
        Degree of the B-spline basis (cubic splines by default).
    family : {"gaussian", "poisson", "binomial"}, default="gaussian"
        Distribution family for the GLM.

    Returns
    -------
    best_alpha : float
        Optimal alpha value.
    best_gam : GAMRegressor
        Fitted GAM with optimal alpha.
    """
    if metric in ("aic", "bic"):
        # AIC/BIC computed on full dataset - parallelize across alphas
        tasks = []
        for alpha in alphas:
            task = delayed(_fit_and_score_ic)(
                alpha, X, y, metric, lower_bound, upper_bound, degree_of_freedom, degree, family
            )
            tasks.append(task)

        results = Parallel(n_jobs=-1)(tasks)

        # Build scores array maintaining alpha order
        alpha_to_score = {}
        for alpha, score in results:
            alpha_to_score[alpha] = score

        scores = np.zeros(len(alphas))
        for i, alpha in enumerate(alphas):
            scores[i] = alpha_to_score[alpha]

        best_alpha = alphas[np.argmin(scores)]

    else:
        # CV metrics - parallelize across all (alpha, fold) combinations
        folds = list(cross_validator.split(X))
        n_folds = len(folds)

        tasks = []
        for alpha in alphas:
            for fold_idx, (train_idx, test_idx) in enumerate(folds):
                task = delayed(_fit_and_score_cv)(
                    alpha,
                    fold_idx,
                    train_idx,
                    test_idx,
                    X,
                    y,
                    metric,
                    lower_bound,
                    upper_bound,
                    degree_of_freedom,
                    degree,
                    family,
                )
                tasks.append(task)

        results = Parallel(n_jobs=-1)(tasks)

        # Reconstruct gridsearch matrix
        gridsearch_matrix = np.zeros((len(alphas), n_folds))
        alpha_to_idx = {}
        for i, alpha in enumerate(alphas):
            alpha_to_idx[alpha] = i

        for alpha, fold_idx, score in results:
            gridsearch_matrix[alpha_to_idx[alpha]][fold_idx] = score

        mean_score = np.mean(gridsearch_matrix, axis=1)
        if metric == "r_squared":
            best_alpha = alphas[np.argmax(mean_score)]
        else:
            best_alpha = alphas[np.argmin(mean_score)]

    best_gam = GAMRegressor(
        alpha=best_alpha,
        degree_of_freedom=degree_of_freedom,
        degree=degree,
        family=family,
    ).fit(X, y, lower_bound=lower_bound, upper_bound=upper_bound)

    return best_alpha, best_gam


def optimize_gam(
    X: np.ndarray,
    y: np.ndarray,
    cross_validator: Optional[BaseCrossValidator] = None,
    alphas: Optional[np.ndarray] = None,
    lower_bound: Optional[float] = None,
    upper_bound: Optional[float] = None,
    bound_padding_ratio: float = 0.1,
    metric: Literal["aic", "bic", "euclidean", "mse", "r_squared"] = "aic",
    degree_of_freedom: int = 20,
    degree: int = 3,
    family: Literal["gaussian", "poisson", "binomial"] = "gaussian",
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
        Lower bound for spline knots. If None, computed as min(X) minus
        padding based on bound_padding_ratio.
    upper_bound : float, optional
        Upper bound for spline knots. If None, computed as max(X) plus
        padding based on bound_padding_ratio.
    bound_padding_ratio : float, default=0.1
        Fraction of the data range to use as padding when computing default
        bounds. For example, 0.1 adds 10% of (max(X) - min(X)) as padding.
    metric : {"aic", "bic", "euclidean", "mse", "r_squared"}, default="aic"
        Metric to optimize. AIC/BIC are computed on full data, others use
        cross-validation. Note: r_squared is maximized, others are minimized.
    degree_of_freedom : int, default=20
        Degrees of freedom for the spline basis.
    degree : int, default=3
        Degree of the B-spline basis (cubic splines by default).
    family : {"gaussian", "poisson", "binomial"}, default="gaussian"
        Distribution family for the GLM.

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
    data_range = float(np.max(X) - np.min(X))
    padding = bound_padding_ratio * data_range

    if lower_bound is None:
        lower_bound = float(np.min(X)) - padding
    if upper_bound is None:
        upper_bound = float(np.max(X)) + padding
    if alphas is None:
        alphas = np.logspace(-4, 4, 100)
    if cross_validator is None:
        cross_validator = choose_cross_validator(X)

    best_alpha, best_gam = optimize_gam_cv(
        X,
        y,
        alphas=alphas,
        cross_validator=cross_validator,
        metric=metric,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        degree_of_freedom=degree_of_freedom,
        degree=degree,
        family=family,
    )

    return best_alpha, best_gam


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
        X-axis values (years).
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
    if draw_historic_timeseries is None:
        raise ImportError('Plotly is required for plotting. Install with: pip install ecoscope["plotting"]')

    df = pd.DataFrame(
        {
            "x": x,
            "y_orig": y_orig,
            "y_mean": y_mean,
            "y_lower": y_lower,
            "y_upper": y_upper,
        }
    )

    return draw_historic_timeseries(
        df,
        current_value_column="y_orig",
        current_value_title="Raw Data",
        historic_min_column="y_lower",
        historic_max_column="y_upper",
        historic_band_title="95% CI",
        historic_mean_column="y_mean",
        historic_mean_title="Trend",
        time_column="x",
        layout_kwargs={"title": plot_title, "xaxis_title": xlabel, "yaxis_title": ylabel},
    )
