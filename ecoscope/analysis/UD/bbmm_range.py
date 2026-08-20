"""Brownian Bridge Movement Model (BBMM) home-range estimation (Horne, Garton,
Krone & Lewis 2007, "Analyzing animal movements using Brownian bridges",
Ecology 88(9):2354-63). Validated against R's CRAN `BBMM` package and
Kranstauber et al. (2012) Eqn 1."""

import logging

import geopandas as gpd  # type: ignore[import-untyped]
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar  # type: ignore[import-untyped]

from ecoscope import Trajectory
from ecoscope.analysis.UD.etd_range import grid_size_from_geographic_extent
from ecoscope.io import raster

logger = logging.getLogger(__name__)


def _extract_points(trajectory_gdf: gpd.GeoDataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Reconstruct the point sequence (xy, t) from a trajectory's LineString
    segments, reusing `Trajectory.to_relocations()` - the same canonical
    trajectory->points conversion `convert_trajectory_to_relocations` uses for
    MCP - rather than a separate reimplementation. `t` is UNIX seconds (float)."""
    relocs_gdf = Trajectory(gdf=trajectory_gdf).to_relocations().gdf
    xy = np.column_stack([relocs_gdf.geometry.x.to_numpy(), relocs_gdf.geometry.y.to_numpy()])
    t = relocs_gdf["fixtime"].astype("int64").to_numpy() / 1e9
    return xy, t


def _bridge_variance(time_lag, alpha, sigma_m2: float, location_error: float):
    """Horne et al. 2007, Eqn 1 (also Kranstauber et al. 2012, Eqn 1)."""
    return time_lag * alpha * (1 - alpha) * sigma_m2 + ((1 - alpha) ** 2 + alpha**2) * location_error**2


def _compute_midpoints(trajectory_gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    """Odd-i leave-one-out test points for the motion-variance MLE (Horne et
    al. 2007, Eqn 1): only every *other* interior fix is used, so consecutive
    leave-one-out tests don't share an overlapping neighbour (which would
    violate the i.i.d. assumption behind the likelihood product)."""
    xy, t = _extract_points(trajectory_gdf)

    z_prev, z_curr, z_next = xy[:-2], xy[1:-1], xy[2:]
    t_prev, t_curr, t_next = t[:-2], t[1:-1], t[2:]

    sl = slice(0, None, 2)
    z_prev, z_curr, z_next = z_prev[sl], z_curr[sl], z_next[sl]
    t_prev, t_curr, t_next = t_prev[sl], t_curr[sl], t_next[sl]

    jump_time = t_next - t_prev
    valid = jump_time > 0
    alpha = np.where(valid, (t_curr - t_prev) / np.where(valid, jump_time, 1.0), np.nan)
    mu = (1 - alpha[:, None]) * z_prev + alpha[:, None] * z_next
    d2 = np.sum((z_curr - mu) ** 2, axis=1)

    midpoints = pd.DataFrame({"jump_time": jump_time, "alpha": alpha, "d2": d2})
    return midpoints[valid & midpoints["alpha"].between(0, 1)].reset_index(drop=True)


def _neg_log_likelihood(sigma_m2: float, midpoints: pd.DataFrame, location_error: float) -> float:
    variance = _bridge_variance(
        midpoints["jump_time"].to_numpy(),
        midpoints["alpha"].to_numpy(),
        sigma_m2,
        location_error,
    )
    d2 = midpoints["d2"].to_numpy()
    log_lik = -np.log(2 * np.pi * variance) - d2 / (2 * variance)
    return -log_lik.sum()


def estimate_motion_variance(trajectory_gdf: gpd.GeoDataFrame, location_error: float) -> float:
    """MLE for a single, whole-track sigma_m^2 (Horne et al. 2007, Eqn 1)."""
    midpoints = _compute_midpoints(trajectory_gdf)
    if midpoints.empty:
        raise ValueError("Not enough interior fixes with valid time gaps to estimate motion variance.")
    result = minimize_scalar(
        _neg_log_likelihood,
        bounds=(1.0, 1_000_000.0),
        method="bounded",
        args=(midpoints, location_error),
    )
    logger.info(f"Estimated Brownian motion variance (sigma_m^2) = {result.x:.2f} m^2/s (from {len(midpoints)} fixes)")
    return float(result.x)


def _build_grid(
    trajectory_gdf: gpd.GeoDataFrame,
    crs: str,
    pixel_size: float,
    expansion_factor: float,
):
    x_min, y_min, x_max, y_max = trajectory_gdf.geometry.total_bounds
    if expansion_factor > 1.0:
        dx = (x_max - x_min) * (expansion_factor - 1.0) / 2.0
        dy = (y_max - y_min) * (expansion_factor - 1.0) / 2.0
        x_min, x_max = x_min - dx, x_max + dx
        y_min, y_max = y_min - dy, y_max + dy

    raster_profile = raster.RasterProfile(
        pixel_size=pixel_size,
        crs=crs,
        nodata_value=np.nan,
        band_count=1,
        raster_extent=raster.RasterExtent(x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max),
    )
    num_rows, num_cols = raster_profile.rows, raster_profile.columns
    col_centers = x_min + pixel_size * (np.arange(num_cols) + 0.5)
    row_centers = y_max - pixel_size * (np.arange(num_rows) + 0.5)
    return raster_profile, col_centers, row_centers


def calculate_bbmm_range(
    trajectory_gdf: gpd.GeoDataFrame,
    crs: str = "EPSG:3857",
    location_error: float = 20.0,
    time_step_seconds: float = 60.0,
    expansion_factor: float = 1.3,
    max_steps_per_segment: int = 50,
    window_padding_sigma: float = 4.0,
    grid_scale_factor: int = 500,
    max_data_gap_seconds: float | None = 14400.0,
) -> raster.RasterData:
    """Estimate a home range using the classic (non-dynamic) Brownian Bridge
    Movement Model, returning the utilization-distribution raster.

    Parameters
    ----------
    trajectory_gdf : gpd.GeoDataFrame
        A trajectory's segment geodataframe (one row per segment, each a
        2-point LineString from `segment_start` to `segment_end`).
    crs : str
        The projected coordinate reference system to compute the surface in.
    location_error : float
        Typical GPS collar accuracy - the standard deviation of a single
        fix's positional error.
    time_step_seconds : float
        How finely each segment's bridge integration is discretized.
    expansion_factor : float
        Pads the calculation grid beyond the trajectory's own bounding box,
        so the density surface has room to taper off naturally instead of
        being cut off at the edge.
    max_steps_per_segment : int
        Caps how finely each segment's bridge integration is discretized,
        regardless of its own time gap.
    window_padding_sigma : float
        Sizes the padded rectangular window each segment's density is
        computed over - a vectorization/performance device, not part of the
        Horne et al. 2007 formula itself.
    grid_scale_factor : int
        Passed to `grid_size_from_geographic_extent` to size the output
        grid's pixels from the trajectory's own geographic extent.
    max_data_gap_seconds : float or None
        Segments with a time gap at or beyond this threshold (default
        14400.0 = 4 hours) are dropped entirely rather than modeled as a
        Brownian bridge - bridge variance scales linearly with time lag, so
        one atypically long gap (e.g. a multi-day collar dropout amid
        otherwise hourly fixes) would otherwise produce a hugely diffuse,
        near-uniform blob dominating the whole surface, even though it
        reflects a data outage rather than genuine movement uncertainty.
        Pass `None` to disable this exclusion entirely.

    Returns
    -------
    raster.RasterData
        The utilization-distribution surface, normalized to sum to 1.
    """
    trajectory_gdf = trajectory_gdf.to_crs(crs)
    sigma_m2 = estimate_motion_variance(trajectory_gdf, location_error)
    pixel_size = grid_size_from_geographic_extent(trajectory_gdf, scale_factor=grid_scale_factor)

    raster_profile, col_centers, row_centers = _build_grid(trajectory_gdf, crs, pixel_size, expansion_factor)
    num_rows, num_cols = raster_profile.rows, raster_profile.columns
    ud = np.zeros((num_rows, num_cols), dtype=np.float64)

    xy, t = _extract_points(trajectory_gdf)
    n_segments = len(xy) - 1

    for i in range(n_segments):
        z0, z1 = xy[i], xy[i + 1]
        time_lag = t[i + 1] - t[i]
        if time_lag <= 0:
            continue
        if max_data_gap_seconds is not None and time_lag >= max_data_gap_seconds:
            continue

        n_steps = min(
            max_steps_per_segment,
            max(2, int(np.ceil(time_lag / time_step_seconds)) + 1),
        )
        alphas = np.linspace(0.0, 1.0, n_steps)
        dt = time_lag / (n_steps - 1)

        variances = _bridge_variance(time_lag, alphas, sigma_m2, location_error)
        max_sigma = np.sqrt(variances.max())
        pad = window_padding_sigma * max_sigma + pixel_size

        seg_x_min, seg_x_max = min(z0[0], z1[0]) - pad, max(z0[0], z1[0]) + pad
        seg_y_min, seg_y_max = min(z0[1], z1[1]) - pad, max(z0[1], z1[1]) + pad

        col_mask = (col_centers >= seg_x_min) & (col_centers <= seg_x_max)
        row_mask = (row_centers >= seg_y_min) & (row_centers <= seg_y_max)
        if not col_mask.any() or not row_mask.any():
            continue

        local_x = col_centers[col_mask]
        local_y = row_centers[row_mask]
        X, Y = np.meshgrid(local_x, local_y)

        local_accum = np.zeros_like(X)
        for alpha, variance in zip(alphas, variances):
            mu_x = (1 - alpha) * z0[0] + alpha * z1[0]
            mu_y = (1 - alpha) * z0[1] + alpha * z1[1]
            ztz = (X - mu_x) ** 2 + (Y - mu_y) ** 2
            local_accum += (1.0 / (2 * np.pi * variance)) * np.exp(-ztz / (2 * variance))

        local_accum *= dt  # Riemann-sum approximation of the time integral over this segment
        row_idx = np.where(row_mask)[0]
        col_idx = np.where(col_mask)[0]
        ud[np.ix_(row_idx, col_idx)] += local_accum

    cell_area = pixel_size * pixel_size
    total_mass = ud.sum() * cell_area
    if total_mass > 0:
        ud /= total_mass

    return raster.RasterData(
        data=ud.astype("float32"),
        crs=raster_profile.crs,
        transform=raster_profile.transform,
    )
