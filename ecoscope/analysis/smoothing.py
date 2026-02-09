import datetime
from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.interpolate import make_interp_spline  # type: ignore[import-untyped]


@dataclass
class SmoothingConfig:
    """
    Configuration for data smoothing.

    Attributes:
    ----------
    method : Literal["spline"]
        The smoothing method to apply. Currently supports "spline".
    y_min : float, optional
        The minimum value to clamp smoothed values to.
        Useful for data like precipitation where values should not go below zero.
    y_max : float, optional
        The maximum value to clamp smoothed values to.
    resolution : int, optional
        The resolution multiplier for interpolation points.
        The number of output points will be len(x) * resolution.
        Default is 10.
    degree : int, optional
        The degree of the spline. Default is 3 (cubic spline).
        - 1: Linear interpolation
        - 2: Quadratic spline
        - 3: Cubic spline (recommended)
        - 4, 5: Higher degree (smoother but may oscillate more)
    """

    method: Literal["spline"]
    y_min: float | None = None
    y_max: float | None = None
    resolution: int = 10
    degree: int = 3


def apply_smoothing(x: np.ndarray, y: np.ndarray, config: SmoothingConfig) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply smoothing to x, y data and return smoothed values.

    Parameters
    ----------
    x: np.ndarray
        The x values
    y: np.ndarray
        The y values
    config: SmoothingConfig
        The smoothing configuration

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        The smoothed (x, y) values
    """

    if config.method != "spline":
        raise ValueError(f"Unsupported smoothing method: {config.method}")

    # Sort for interpolation
    sort_idx = np.argsort(x)
    x_sorted, y_sorted = x[sort_idx], y[sort_idx]

    # Check for datetime types (numpy datetime64 or Python date/datetime objects)
    is_numpy_datetime = np.issubdtype(x_sorted.dtype, "datetime64")
    is_python_date = len(x_sorted) > 0 and isinstance(x_sorted[0], (datetime.date, datetime.datetime))

    # Convert to numeric for interpolation
    if is_numpy_datetime:
        x_numeric = x_sorted.astype(np.int64)
    elif is_python_date:
        # Convert Python date/datetime to numpy datetime64, then to int64
        x_sorted = np.array(x_sorted, dtype="datetime64[ns]")
        x_numeric = x_sorted.astype(np.int64)
    else:
        x_numeric = x_sorted.astype(np.float64)

    # Interpolate with spline
    x_smooth = np.linspace(x_numeric.min(), x_numeric.max(), len(x) * config.resolution)
    spline = make_interp_spline(x_numeric, y_sorted, k=config.degree)
    y_smooth = spline(x_smooth)

    # Clamp to y_min/y_max if specified
    if config.y_min is not None:
        y_smooth = np.maximum(y_smooth, config.y_min)
    if config.y_max is not None:
        y_smooth = np.minimum(y_smooth, config.y_max)

    # Convert x back to datetime if needed
    if is_numpy_datetime or is_python_date:
        x_smooth = x_smooth.astype("datetime64[ns]")

    return x_smooth, y_smooth
