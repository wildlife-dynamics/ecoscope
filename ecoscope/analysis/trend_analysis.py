"""
Trend Analysis using Generalized Additive Models (GAMs).

This module provides tools for fitting three regression patterns: linear
regression (OLS), generalized additive model (GAM), and generalized additive
mixed model (GAMM) to time series data, particularly useful for analyzing
environmental trends from remote sensing data. GAM and GAMM are both fit via
Bayesian inference (PyMC, through Bambi) - GAMM adds a per-site random effect
on top of GAM's shared smooth trend (see GAMMRegressor's docstring).
"""

from typing import Literal, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import statsmodels.api as sm  # type: ignore[import-not-found,import-untyped]
    from sklearn.base import BaseEstimator, RegressorMixin  # type: ignore[import-not-found,import-untyped]
except ModuleNotFoundError:
    raise ModuleNotFoundError(
        'Missing optional dependencies required by this module. Please run pip install ecoscope["trends"]'
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

    def summary(self) -> pd.DataFrame:
        """Fit parameter summary (coefficient, std error, p-value, 95% CI)
        from the underlying statsmodels fit - the direct output of its own
        `.fit()` call. Only used by LinearRegressionRegressor; overridden by
        GAMRegressor and GAMMRegressor, whose fits are Bayesian (no
        p-values/statsmodels result to summarize this way)."""
        self._check_is_fitted()
        res = self._res_
        params = np.asarray(res.params)
        names = list(getattr(res.model, "exog_names", None) or [f"param_{i}" for i in range(len(params))])
        conf_int = np.asarray(res.conf_int())
        return pd.DataFrame(
            {
                "parameter": names,
                "coefficient": params,
                "std_error": np.asarray(res.bse),
                "p_value": np.asarray(res.pvalues),
                "ci_lower": conf_int[:, 0],
                "ci_upper": conf_int[:, 1],
            }
        )


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
    random_seed : int, optional
        Seed for the MCMC sampler, for reproducible fits (``mcmc`` only).
    """

    def __init__(
        self,
        degree_of_freedom: int = 10,
        inference_method: Literal["mcmc", "laplace"] = "mcmc",
        draws: int = 500,
        tune: Optional[int] = None,
        chains: int = 2,
        family: str = "gaussian",
        random_seed: Optional[int] = None,
    ):
        if inference_method not in ("mcmc", "laplace"):
            raise ValueError(f"Unsupported inference_method: {inference_method!r}. " 'Must be "mcmc" or "laplace".')
        self.degree_of_freedom = degree_of_freedom
        self.inference_method = inference_method
        self.draws = draws
        self.tune = tune
        self.chains = chains
        self.family = family
        self.random_seed = random_seed

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
                'Please run pip install ecoscope["trends"]'
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
                random_seed=self.random_seed,
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

    def summary(self) -> pd.DataFrame:
        """Posterior parameter summary (mean, sd, hdi_3%, hdi_97%, ess_bulk,
        ess_tail, r_hat per parameter) from the fitted MCMC/Laplace trace -
        the direct output of pymc's own fit, via ArviZ."""
        self._check_is_fitted()
        import arviz as az  # type: ignore[import-not-found,import-untyped]

        return az.summary(self._idata_).reset_index(names="parameter")


class GAMRegressor(_TrendRegressorBase):
    """
    Generalized Additive Model (GAM) Regressor using Bambi (Bayesian).

    Fits a smoothed B-spline trend via PyMC/Bambi - the same machinery as
    GAMMRegressor, but with no per-site random effect: each call fits and
    predicts independently (see GAMMRegressor's docstring for the combined,
    multi-site fit). Provides genuine posterior credible intervals rather
    than frequentist confidence intervals, at the cost of MCMC sampling.

    Parameters
    ----------
    degree_of_freedom : int, default=10
        Degrees of freedom for the spline basis
    inference_method : {"mcmc", "laplace"}, default="mcmc"
        Inference method. ``"mcmc"`` is the reliable default for spline
        models. ``"laplace"`` is faster when it converges but may fail for
        some model specifications.
    draws : int, default=500
        Number of posterior samples (``mcmc`` only).
    tune : int, optional
        Number of tuning steps for MCMC. Defaults to ``draws``.
    chains : int, default=2
        Number of MCMC chains (``mcmc`` only).
    family : str, default="gaussian"
        Response distribution family. Supports "gaussian", "poisson",
        "gamma", "bernoulli".
    random_seed : int, optional
        Seed for the MCMC sampler, for reproducible fits (``mcmc`` only).

    Examples
    --------
    >>> from ecoscope.analysis.trend_analysis import GAMRegressor
    >>> import numpy as np
    >>> X = np.array([2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009])
    >>> y = np.array([100, 95, 90, 85, 80, 75, 70, 65, 60, 55])
    >>> gam = GAMRegressor(draws=200, chains=1).fit(X, y)
    >>> predictions = gam.predict(X)
    """

    def __init__(
        self,
        degree_of_freedom: int = 10,
        inference_method: Literal["mcmc", "laplace"] = "mcmc",
        draws: int = 500,
        tune: Optional[int] = None,
        chains: int = 2,
        family: str = "gaussian",
        random_seed: Optional[int] = None,
    ):
        if inference_method not in ("mcmc", "laplace"):
            raise ValueError(f"Unsupported inference_method: {inference_method!r}. " 'Must be "mcmc" or "laplace".')
        self.degree_of_freedom = degree_of_freedom
        self.inference_method = inference_method
        self.draws = draws
        self.tune = tune
        self.chains = chains
        self.family = family
        self.random_seed = random_seed

    def fit(self, X, y):
        """
        Fit the GAM model.

        Parameters
        ----------
        X : array-like of shape (n_samples,)
            Training years (or other time index).
        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        self : GAMRegressor
            Returns self for method chaining.
        """
        try:
            import bambi as bmb  # type: ignore[import-not-found,import-untyped]
        except ModuleNotFoundError as err:
            raise ModuleNotFoundError(
                "Missing optional dependency bambi required by GAMRegressor. "
                'Please run pip install ecoscope["trends"]'
            ) from err

        # Scale y only for gaussian (poisson etc. need non-negative y)
        X_norm, y_norm, self._X_min_, self._y_mean_, self._y_std_ = _normalize_domain(
            X, y, standardize_y=self.family == "gaussian"
        )
        X_norm = X_norm.ravel()

        self._df_ = pd.DataFrame({"year": X_norm, "y": y_norm})

        self._model_ = bmb.Model(
            f"y ~ bs(year, df={self.degree_of_freedom})",
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
                random_seed=self.random_seed,
            )

        return self

    def predict(self, X):
        """
        Predict the mean trend.

        Parameters
        ----------
        X : array-like of shape (n_samples,)
            Years (same scale as in fit).

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
        pred_df = pd.DataFrame({"year": X_norm})

        fitted = self._model_.predict(self._idata_, data=pred_df, kind="response_params", inplace=False)
        y_norm_pred = fitted.posterior["mu"].mean(dim=["chain", "draw"]).values
        return y_norm_pred * self._y_std_ + self._y_mean_

    def predict_with_ci(self, X, credible_mass=0.95) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict with Bayesian credible intervals.

        Parameters
        ----------
        X : array-like of shape (n_samples,)
            Years (same scale as in ``fit``).
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
        pred_df = pd.DataFrame({"year": X_norm})

        fitted = self._model_.predict(self._idata_, data=pred_df, kind="response_params", inplace=False)

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
        raise NotImplementedError("GAM is Bayesian; use waic() or loo() instead of aic().")

    def bic(self) -> float:
        raise NotImplementedError("GAM is Bayesian; use waic() or loo() instead of bic().")

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

    def summary(self) -> pd.DataFrame:
        """Posterior parameter summary (mean, sd, hdi_3%, hdi_97%, ess_bulk,
        ess_tail, r_hat per parameter) from the fitted MCMC/Laplace trace -
        the direct output of pymc's own fit, via ArviZ."""
        self._check_is_fitted()
        import arviz as az  # type: ignore[import-not-found,import-untyped]

        return az.summary(self._idata_).reset_index(names="parameter")


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
