"""
Trend Analysis using Generalized Additive Models (GAMs).

This module provides tools for fitting four regression patterns: linear regression,
generalized linear model (GLM), generalized additive model (GAM), and generalized
additive mixed model (GAMM) to time series data, particularly useful for analyzing
environmental trends from remote sensing data.
"""

from typing import Literal, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import statsmodels.api as sm  # type: ignore[import-not-found,import-untyped]
    from scipy.spatial.distance import euclidean  # type: ignore[import-not-found,import-untyped]
    from sklearn.base import BaseEstimator, RegressorMixin  # type: ignore[import-not-found,import-untyped]
    from sklearn.model_selection import (  # type: ignore[import-not-found,import-untyped]
        BaseCrossValidator,
        LeaveOneOut,
        TimeSeriesSplit,
    )
    from statsmodels.gam.api import BSplines, GLMGam  # type: ignore[import-not-found,import-untyped]
    from statsmodels.genmod.families import (  # type: ignore[import-not-found,import-untyped]
        Binomial,
        Gamma,
        Gaussian,
        Poisson,
    )
    from statsmodels.genmod.families.links import Log  # type: ignore[import-untyped]
except ModuleNotFoundError:
    raise ModuleNotFoundError(
        'Missing optional dependencies required by this module. Please run pip install ecoscope["analysis"]'
    )


def _normalize_domain(X, y, *, standardize_y: bool = True, x_offset: Optional[float] = None):
    """
    Prepare X and y for fitting.

    - X: subtract min (or x_offset if given) so years start at 0.
    - y: if standardize_y, center and scale to mean 0 / std 1; otherwise leave as-is.

    Returns X_norm, y_norm, and the values needed to undo the transforms later.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    y = np.asarray(y, dtype=float).ravel()

    if x_offset is not None:
        X_min = float(x_offset)
    else:
        X_min = float(X.min())
    X_norm = X - X_min

    if standardize_y:
        y_mean = float(y.mean())
        y_std = float(y.std()) or 1.0
        y_norm = (y - y_mean) / y_std
    else:
        # No y scaling: store 0/1 so predict can still do y * std + mean
        y_mean, y_std, y_norm = 0.0, 1.0, y

    return X_norm, y_norm, X_min, y_mean, y_std


class _TrendRegressorBase(BaseEstimator, RegressorMixin):
    """Shared metrics for trend regressors."""

    def _check_is_fitted(self) -> None:
        if not hasattr(self, "_res_") and not hasattr(self, "_idata_"):
            raise ValueError("Model has not been fitted. Call fit() before using this method.")

    def r_squared(self, X, y, **predict_kwargs) -> float:
        self._check_is_fitted()
        y_pred = self.predict(X, **predict_kwargs)
        y = np.asarray(y).ravel()
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        if ss_tot == 0:
            if ss_res == 0:
                return 1.0
            return 0.0
        return float(1 - ss_res / ss_tot)

    def mse(self, X, y, **predict_kwargs) -> float:
        self._check_is_fitted()
        y_pred = self.predict(X, **predict_kwargs)
        y = np.asarray(y).ravel()
        return float(np.mean((y - y_pred) ** 2))

    def aic(self) -> float:
        """Return Akaike Information Criterion."""
        self._check_is_fitted()
        return float(self._res_.aic)

    def bic(self) -> float:
        """Return Bayesian Information Criterion."""
        self._check_is_fitted()

        bic_llf = getattr(self._res_, "bic_llf", None)
        if bic_llf is not None:
            return float(bic_llf)
        return float(self._res_.bic)


class GAMMRegressor(_TrendRegressorBase):
    """
    Generalized Additive Mixed Model (GAMM) Regressor using Bambi.

    Fits a GAM with site-level random effects using Bayesian inference.
    Provides genuine posterior credible intervals rather than frequentist
    confidence intervals.

    Parameters
    ----------
    degree_of_freedom : int, default=10
        Degrees of freedom for the spline basis
    inference_method : {"mcmc", "laplace"}, default="mcmc"
        Inference method. ``"mcmc"`` is the reliable default for spline
        models with random effects. ``"laplace"`` is faster when it
        converges but may fail for some model specifications.
    draws : int, default=500
        Number of posterior samples (``mcmc`` only).
    tune : int, optional
        Number of tuning steps for MCMC. Defaults to ``draws``.
    chains : int, default=2
        Number of MCMC chains (``mcmc`` only).
    family : str, default="gaussian"
        Response distribution family. Supports "gaussian", "poisson",
        "gamma", "bernoulli".
    """

    def __init__(
        self,
        degree_of_freedom: int = 10,
        inference_method: Literal["mcmc", "laplace"] = "mcmc",
        draws: int = 500,
        tune: Optional[int] = None,
        chains: int = 2,
        family: str = "gaussian",
    ):
        if inference_method not in ("mcmc", "laplace"):
            raise ValueError(f"Unsupported inference_method: {inference_method!r}. " 'Must be "mcmc" or "laplace".')
        self.degree_of_freedom = degree_of_freedom
        self.inference_method = inference_method
        self.draws = draws
        self.tune = tune
        self.chains = chains
        self.family = family

    def fit(self, X, y, site_ids):
        """
        Fit the GAMM model.

        Parameters
        ----------
        X : array-like of shape (n_samples,)
            Training years (or other time index).
        y : array-like of shape (n_samples,)
            Target values.
        site_ids : array-like of shape (n_samples,)
            Site label for each observation.

        Returns
        -------
        self : GAMMRegressor
            Returns self for method chaining.
        """
        try:
            import bambi as bmb  # type: ignore[import-not-found,import-untyped]
        except ModuleNotFoundError as err:
            raise ModuleNotFoundError(
                "Missing optional dependency bambi required by GAMMRegressor. "
                'Please run pip install ecoscope["gamm"]'
            ) from err

        site_ids = np.asarray(site_ids).ravel()
        # Scale y only for gaussian (poisson etc. need non-negative y)
        X_norm, y_norm, self._X_min_, self._y_mean_, self._y_std_ = _normalize_domain(
            X, y, standardize_y=self.family == "gaussian"
        )
        X_norm = X_norm.ravel()

        # Build DataFrame for Bambi
        self._df_ = pd.DataFrame(
            {
                "year": X_norm,
                "y": y_norm,
                "site_id": site_ids,
            }
        )

        # Fit GAMM
        self._model_ = bmb.Model(
            f"y ~ bs(year, df={self.degree_of_freedom}) + (1|site_id)",
            self._df_,
            family=self.family,
        )

        if self.inference_method == "laplace":
            self._idata_ = self._model_.fit(inference_method="laplace")
        else:
            if self.tune is not None:
                tune = self.tune
            else:
                tune = self.draws
            self._idata_ = self._model_.fit(
                draws=self.draws,
                tune=tune,
                chains=self.chains,
            )

        return self

    def predict(self, X, site_ids=None):
        """
        Predict the mean trend.

        Parameters
        ----------
        X : array-like of shape (n_samples,)
            Years (same scale as in fit).
        site_ids : array-like of shape (n_samples,), optional
            Site label per row. When provided, predictions include each
            site's random intercept (site-specific trend level). When
            omitted, predictions use only the global ``bs(year)`` smooth —
            the average trend across sites, with all site random effects
            set to zero.

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
        X = np.asarray(X).ravel()
        X_norm = X - self._X_min_
        include_group_specific = site_ids is not None

        if site_ids is not None:
            pred_df = pd.DataFrame(
                {
                    "year": X_norm,
                    "site_id": np.asarray(site_ids).ravel(),
                }
            )
        else:
            pred_df = pd.DataFrame(
                {
                    "year": X_norm,
                    "site_id": np.full(len(X), self._df_["site_id"].iloc[0]),
                }
            )

        fitted = self._model_.predict(
            self._idata_,
            data=pred_df,
            kind="response_params",
            include_group_specific=include_group_specific,
            inplace=False,
        )
        y_norm_pred = fitted.posterior["mu"].mean(dim=["chain", "draw"]).values
        return y_norm_pred * self._y_std_ + self._y_mean_

    def predict_with_ci(self, X, site_ids=None, credible_mass=0.95) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict with Bayesian credible intervals.

        Parameters
        ----------
        X : array-like of shape (n_samples,)
            Years (same scale as in ``fit``).
        site_ids : array-like of shape (n_samples,), optional
            Same semantics as :meth:`predict`. Pass site labels for
            site-specific intervals; omit for population-level intervals.
        credible_mass : float, default=0.95
            Width of the credible interval. 0.95 means there is a 95%
            probability the true mean trend lies within the returned bounds.

        Returns
        -------
        mean : ndarray
            Predicted mean values.
        ci_lower : ndarray
            Lower bound of the credible interval.
        ci_upper : ndarray
            Upper bound of the credible interval.

        Raises
        ------
        ValueError
            If the model has not been fitted.
        """
        self._check_is_fitted()

        X = np.asarray(X).ravel()
        X_norm = X - self._X_min_
        include_group_specific = site_ids is not None

        if site_ids is not None:
            pred_df = pd.DataFrame(
                {
                    "year": X_norm,
                    "site_id": np.asarray(site_ids).ravel(),
                }
            )
        else:
            pred_df = pd.DataFrame(
                {
                    "year": X_norm,
                    "site_id": np.full(len(X), self._df_["site_id"].iloc[0]),
                }
            )

        fitted = self._model_.predict(
            self._idata_,
            data=pred_df,
            kind="response_params",
            include_group_specific=include_group_specific,
            inplace=False,
        )

        samples = fitted.posterior["mu"].values
        samples_flat = samples.reshape(-1, samples.shape[-1])

        mean = samples_flat.mean(axis=0)
        lower = np.percentile(samples_flat, (1 - credible_mass) / 2 * 100, axis=0)
        upper = np.percentile(samples_flat, (1 + credible_mass) / 2 * 100, axis=0)

        return (
            mean * self._y_std_ + self._y_mean_,
            lower * self._y_std_ + self._y_mean_,
            upper * self._y_std_ + self._y_mean_,
        )

    def aic(self) -> float:
        raise NotImplementedError("GAMM is Bayesian; use waic() or loo() instead of aic().")

    def bic(self) -> float:
        raise NotImplementedError("GAMM is Bayesian; use waic() or loo() instead of bic().")

    def waic(self):
        """Return ArviZ WAIC (Watanabe–Akaike information criterion)."""
        self._check_is_fitted()
        import arviz as az  # type: ignore[import-not-found,import-untyped]

        return az.waic(self._idata_)

    def loo(self):
        """Return ArviZ LOO (leave-one-out cross-validation)."""
        self._check_is_fitted()
        import arviz as az  # type: ignore[import-not-found,import-untyped]

        return az.loo(self._idata_)


class GAMRegressor(_TrendRegressorBase):
    """
    Generalized Additive Model (GAM) Regressor using B-Splines.

    A scikit-learn compatible wrapper around statsmodels GLMGam that provides
    a user-friendly interface for fitting GAMs to time series data.

    Parameters
    ----------
    alpha : float or None, default=None
        Smoothing parameter. If None, alpha is selected automatically via
        cross-validation during fit(). Higher values result in smoother curves.
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
    >>> X = np.array([2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009])
    >>> y = np.array([100, 95, 90, 85, 80, 75, 70, 65, 60, 55])
    >>> gam = GAMRegressor().fit(X, y)
    >>> predictions = gam.predict(X)
    """

    def __init__(
        self,
        alpha: Optional[float] = None,
        degree_of_freedom: int = 20,
        degree: int = 3,
        family: Literal["gaussian", "poisson", "binomial"] = "gaussian",
    ):
        self.alpha = alpha
        self.degree_of_freedom = degree_of_freedom
        self.degree = degree
        self._family_name = family

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
        bound_padding_ratio: float = 0.1,
        metric: Literal["aic", "bic", "euclidean", "mse", "r_squared"] = "aic",
        alphas: Optional[np.ndarray] = None,
        cross_validator: Optional["BaseCrossValidator"] = None,
        x_offset: Optional[float] = None,
    ):
        """
        Fit the GAM model.

        When alpha is None (the default), the optimal smoothing parameter is
        selected automatically via cross-validation before fitting.

        Parameters
        ----------
        X : array-like of shape (n_samples, 1) or (n_samples,)
            Training data (typically time/date values).
        y : array-like of shape (n_samples,)
            Target values.
        upper_bound : float, optional
            Upper bound for spline knots. If None, computed from data range.
        lower_bound : float, optional
            Lower bound for spline knots. If None, computed from data range.
        bound_padding_ratio : float, default=0.1
            Fraction of the data range added as padding when computing default bounds.
        metric : {"aic", "bic", "euclidean", "mse", "r_squared"}, default="aic"
            Metric used to select alpha. Only used when alpha=None.
        alphas : ndarray, optional
            Alpha values to search over. Defaults to logspace(-6, 4, 100).
        cross_validator : BaseCrossValidator, optional
            Cross-validation strategy. Chosen automatically if None.
        x_offset : float, optional
            Internal. Value subtracted from X; bounds are assumed to already
            be in the shifted coordinate space when this is set.

        Returns
        -------
        self : GAMRegressor
        """
        X_arr = np.asarray(X).ravel()
        y_arr = np.asarray(y).ravel()

        if x_offset is not None:
            X_min = float(x_offset)
        else:
            X_min = float(X_arr.min())
        data_range = float(np.max(X_arr - X_min))
        padding = bound_padding_ratio * data_range

        if x_offset is not None:
            if lower_bound is not None:
                lb = lower_bound
            else:
                lb = -padding
            if upper_bound is not None:
                ub = upper_bound
            else:
                ub = data_range + padding
        else:
            if lower_bound is not None:
                lb = float(lower_bound) - X_min
            else:
                lb = -padding
            if upper_bound is not None:
                ub = float(upper_bound) - X_min
            else:
                ub = data_range + padding

        if self.alpha is None:
            if alphas is None:
                alphas = np.logspace(-6, 4, 100)
            if cross_validator is None:
                # Small series: LOO. Larger: time-ordered splits
                if len(X_arr) <= 10:
                    cross_validator = LeaveOneOut()
                else:
                    cross_validator = TimeSeriesSplit(n_splits=5)
            self.alpha_ = self._find_best_alpha(X_arr, y_arr, lb, ub, X_min, metric, alphas, cross_validator)
        else:
            self.alpha_ = float(self.alpha)

        X_norm, y_norm, self._X_min_, self._y_mean_, self._y_std_ = _normalize_domain(
            X_arr, y_arr, standardize_y=isinstance(self.family, Gaussian), x_offset=X_min
        )
        knot_kwds: list[dict[str, float]] = [{"upper_bound": ub, "lower_bound": lb}]
        self._spline_ = BSplines(X_norm, df=[self.degree_of_freedom], degree=[self.degree], knot_kwds=knot_kwds)
        exog = np.ones((len(X_norm), 1))
        self._res_ = GLMGam(y_norm, exog=exog, smoother=self._spline_, alpha=self.alpha_, family=self.family).fit()

        return self

    def _find_best_alpha(self, X, y, lb, ub, X_min, metric, alphas, cross_validator) -> float:
        """Grid search for the optimal smoothing parameter."""

        def _score_ic(a):
            gam = GAMRegressor(
                alpha=a, degree_of_freedom=self.degree_of_freedom, degree=self.degree, family=self._family_name
            )
            gam.fit(X, y, lower_bound=lb, upper_bound=ub, x_offset=X_min)
            if metric == "aic":
                return gam.aic()
            return gam.bic()

        def _score_cv(a, train_idx, test_idx):
            gam = GAMRegressor(
                alpha=a, degree_of_freedom=self.degree_of_freedom, degree=self.degree, family=self._family_name
            )
            gam.fit(X[train_idx], y[train_idx], lower_bound=lb, upper_bound=ub, x_offset=X_min)
            X_test, y_test = X[test_idx], y[test_idx]
            if metric == "euclidean":
                return euclidean(y_test, gam.predict(X_test))
            elif metric == "mse":
                return gam.mse(X_test, y_test)
            else:  # r_squared
                return gam.r_squared(X_test, y_test)

        if metric in ("aic", "bic"):
            scores = [_score_ic(a) for a in alphas]
            return float(alphas[np.argmin(scores)])

        folds = list(cross_validator.split(X))
        gridsearch = np.zeros((len(alphas), len(folds)))
        for i, a in enumerate(alphas):
            for fi, (ti, vi) in enumerate(folds):
                gridsearch[i, fi] = _score_cv(a, ti, vi)
        mean_scores = np.mean(gridsearch, axis=1)
        if metric == "r_squared":
            return float(alphas[np.argmax(mean_scores)])
        return float(alphas[np.argmin(mean_scores)])

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

        # Apply X normalization
        X = X - self._X_min_

        exog = np.ones((len(X), 1))
        y_norm = self._res_.predict(exog=exog, exog_smooth=X)

        # Invert y normalization
        return y_norm * self._y_std_ + self._y_mean_

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

        # Apply X normalization
        X = X - self._X_min_

        exog = np.ones((len(X), 1))
        sf = self._res_.get_prediction(exog=exog, exog_smooth=X).summary_frame()

        # Invert all three arrays
        mean = sf["mean"].to_numpy() * self._y_std_ + self._y_mean_
        lower = sf["mean_ci_lower"].to_numpy() * self._y_std_ + self._y_mean_
        upper = sf["mean_ci_upper"].to_numpy() * self._y_std_ + self._y_mean_

        return mean, lower, upper


class GLMRegressor(_TrendRegressorBase):
    """
    Generalized Linear Model (GLM) Regressor.

    Parameters
    ----------
    family : {"gaussian", "poisson", "binomial", "gamma"}, default="gaussian"
        Distribution family for the GLM.
    add_intercept : bool, default=True
        Whether to include an intercept term in the model.
    """

    def __init__(
        self,
        family: Literal["gaussian", "poisson", "binomial", "gamma"] = "gaussian",
        add_intercept: bool = True,
    ):
        self.add_intercept = add_intercept

        if family == "gaussian":
            self.family = Gaussian(link=Log())
        elif family == "poisson":
            self.family = Poisson()
        elif family == "binomial":
            self.family = Binomial()
        elif family == "gamma":
            self.family = Gamma()
        else:
            raise ValueError(f"Unsupported family: {family}. Must be 'gaussian', 'poisson', 'binomial', or 'gamma'")

    def fit(self, X, y):
        # Shift X only; keep y as-is (log/poisson/gamma can't use negative y)
        X, _, self._X_min_, _, _ = _normalize_domain(X, y)
        y = np.asarray(y, dtype=float).ravel()
        self._y_mean_ = 0.0
        self._y_std_ = 1.0

        exog = X
        if self.add_intercept:
            exog = sm.add_constant(exog, has_constant="add")

        self._res_ = sm.GLM(y, exog, family=self.family).fit()
        return self

    def predict(self, X):
        self._check_is_fitted()
        X = np.asarray(X)
        if X.ndim == 1:
            X = X[:, None]

        X = X - self._X_min_
        exog = X
        if self.add_intercept:
            exog = sm.add_constant(exog, has_constant="add")

        y_norm = self._res_.predict(exog=exog)
        return y_norm * self._y_std_ + self._y_mean_

    def predict_with_ci(self, X) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        self._check_is_fitted()
        X = np.asarray(X)
        if X.ndim == 1:
            X = X[:, None]

        X = X - self._X_min_
        exog = X
        if self.add_intercept:
            exog = sm.add_constant(exog, has_constant="add")

        sf = self._res_.get_prediction(exog=exog).summary_frame()
        mean = sf["mean"].to_numpy() * self._y_std_ + self._y_mean_
        lower = sf["mean_ci_lower"].to_numpy() * self._y_std_ + self._y_mean_
        upper = sf["mean_ci_upper"].to_numpy() * self._y_std_ + self._y_mean_

        return mean, lower, upper


class LinearRegressionRegressor(_TrendRegressorBase):
    """
    Standard Linear Regression (OLS) Regressor.

    Parameters
    ----------
    add_intercept : bool, default=True
        Whether to include an intercept term in the model.
    """

    def __init__(self, add_intercept: bool = True):
        self.add_intercept = add_intercept

    def fit(self, X, y):
        X, y, self._X_min_, self._y_mean_, self._y_std_ = _normalize_domain(X, y)

        exog = X
        if self.add_intercept:
            exog = sm.add_constant(exog, has_constant="add")

        self._res_ = sm.OLS(y, exog).fit()
        return self

    def predict(self, X):
        self._check_is_fitted()
        X = np.asarray(X)
        if X.ndim == 1:
            X = X[:, None]

        X = X - self._X_min_
        exog = X
        if self.add_intercept:
            exog = sm.add_constant(exog, has_constant="add")

        y_norm = self._res_.predict(exog)
        return y_norm * self._y_std_ + self._y_mean_

    def predict_with_ci(self, X) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        self._check_is_fitted()
        X = np.asarray(X)
        if X.ndim == 1:
            X = X[:, None]

        X = X - self._X_min_
        exog = X
        if self.add_intercept:
            exog = sm.add_constant(exog, has_constant="add")

        sf = self._res_.get_prediction(exog).summary_frame()
        mean = sf["mean"].to_numpy() * self._y_std_ + self._y_mean_
        lower = sf["mean_ci_lower"].to_numpy() * self._y_std_ + self._y_mean_
        upper = sf["mean_ci_upper"].to_numpy() * self._y_std_ + self._y_mean_

        return mean, lower, upper
