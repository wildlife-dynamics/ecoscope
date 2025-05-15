import logging
import math
import os
import typing
from dataclasses import dataclass

import numpy as np
import geopandas as gpd  # type: ignore[import-untyped]

from ecoscope import Trajectory
from ecoscope.base.utils import BoundingBox
from ecoscope.io import raster

try:
    import numba as nb  # type: ignore[import-untyped]
    import scipy  # type: ignore[import-untyped]
    from scipy.optimize import minimize  # type: ignore[import-untyped]
    from scipy.stats import weibull_min  # type: ignore[import-untyped]
    from sklearn import neighbors  # type: ignore[import-untyped]
    from sklearn.metrics.pairwise import haversine_distances  # type: ignore[import-untyped]
except ModuleNotFoundError:
    raise ModuleNotFoundError(
        'Missing optional dependencies required by this module. \
         Please run pip install ecoscope["analysis"]'
    )

logger = logging.getLogger(__name__)


@nb.cfunc("double(intc, CPointer(double))")
def __etd__(_, a):
    s = a[0]
    k = a[1]
    el = a[2]
    m = a[3]

    return np.exp(-((s / el) ** k)) * s ** (k - 2) / np.sqrt(s**2 - m**2)


_etd = scipy.LowLevelCallable(__etd__.ctypes)


class WeibullPDF:
    @staticmethod
    def fit(data, floc=0):
        # returns estimates of shape and scale parameters from data; keeping the location fixed
        shape, _, scale = weibull_min.fit(data, floc=floc)
        return shape, scale

    @staticmethod
    def pdf(data, shape, location=0, scale=1):
        # probability density function.
        return weibull_min.pdf(x=data, c=shape, loc=location, scale=scale)

    @staticmethod
    def cdf(data, shape, location=0, scale=1):
        # cumulative distribution function.
        return weibull_min.cdf(x=data, c=shape, loc=location, scale=scale)

    @staticmethod
    def nelder_mead(func, x0, args=(), **kwargs):
        # minimization of scalar function of one or more variables using the Nelder-Mead algorithm.
        return minimize(fun=func, x0=x0, args=args, method="Nelder-Mead", **kwargs)

    @staticmethod
    def expected_func(speed, shape, scale, time, distance):
        # time-density expectation function for two-parameter weibull distribution.
        _funcs = [
            4 * shape / (math.pi * scale * speed),
            math.pow((speed / scale), shape - 1),
            math.exp(-1 * math.pow(speed / scale, shape))
            / math.sqrt(np.square(speed) * np.square(time) - np.square(distance)),
        ]
        return math.prod(_funcs)


@dataclass
class Weibull2Parameter(WeibullPDF):
    shape: float = 1.0
    scale: float = 1.0


@dataclass
class Weibull3Parameter(WeibullPDF):
    # Weibull parameterization using scale=a(t^b)(c^t)
    shape: float = 1.0
    a: float = 1.0
    b: float = 1.0
    c: float = 1.0


def calculate_etd_range(
    trajectory: Trajectory | gpd.GeoDataFrame,
    raster_profile: raster.RasterProfile,
    output_path: typing.Union[str, bytes, os.PathLike, None] = None,
    max_speed_kmhr: float = 0.0,
    max_speed_percentage: float = 0.9999,
    expansion_factor: float = 1.3,
    weibull_pdf: Weibull2Parameter = Weibull2Parameter(),
    grid_threshold: int = 100,
) -> raster.RasterData:
    """
    The ETDRange class provides a trajectory-based, nonparametric approach to estimate the utilization distribution (UD)
    of an animal, using model parameters derived directly from the movement behaviour of the species.
    The model builds on the theory of "time-geography" whereby elliptical constrain- ing regions are established
    between temporally adjacent recorded locations.

    Parameters
    ----------
    trajectory_gdf : geopandas.GeoDataFrame
    output_path : str or PathLike or None
    max_speed_kmhr : float
    max_speed_percentage : 0.999
    raster_profile : raster.RasterProfile
    expansion_factor : float
    weibull_pdf : Weibull2Parameter or Weibull3Parameter
    grid_threshold: int

    Returns
    -------
    output_path : str
    """

    trajectory_gdf = trajectory.gdf if isinstance(trajectory, Trajectory) else trajectory
    # if two-parameter weibull has default values; run an optimization routine to auto-determine parameters
    if isinstance(weibull_pdf, Weibull2Parameter) and all([weibull_pdf.shape == 1.0, weibull_pdf.scale == 1.0]):
        speed_kmhr = trajectory_gdf.speed_kmhr
        shape, scale = weibull_pdf.fit(speed_kmhr)

        # update the shape/scale parameters
        weibull_pdf.shape = shape
        weibull_pdf.scale = scale

    # reproject trajseg to desired crs
    trajectory_gdf = trajectory_gdf.to_crs(raster_profile.crs)

    # determine envelope of trajectory
    x_min, y_min, x_max, y_max = trajectory_gdf.geometry.total_bounds

    # apply expansion factor on the trajectory total bound.
    if expansion_factor > 1.0:
        dx = (x_max - x_min) * (expansion_factor - 1.0) / 2.0
        dy = (y_max - y_min) * (expansion_factor - 1.0) / 2.0
        x_min -= dx
        x_max += dx
        y_min -= dy
        y_max += dy

    # update the raster extent for the raster profile
    raster_profile.raster_extent = raster.RasterExtent(x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max)

    # determine the output raster size
    num_rows, num_columns = (
        raster_profile.rows,
        raster_profile.columns,
    )

    # maximum trajectory segment speed value
    max_trajseg_speed = trajectory_gdf.speed_kmhr.max()

    # determine max-speed value
    if max_speed_kmhr > 0.0:
        maxspeed = max_speed_kmhr
    else:
        # Use a value calculated from the CDF
        maxspeed = weibull_pdf.scale * math.pow(
            -1 * math.log(1.0 - max_speed_percentage),
            1.0 / weibull_pdf.shape,
        )
    if maxspeed <= max_trajseg_speed:
        raise ValueError(
            f"ETD maximum speed value: {maxspeed} should be greater than "
            f"trajectory maximum speed value {max_trajseg_speed}"
        )

    # Define the affine transform to get grid pixel centroids as geographic coordinates.
    grid_centroids = np.array(raster_profile.transform.to_gdal()).reshape(2, 3)
    grid_centroids[0, 0] = x_min + raster_profile.pixel_size * 0.5
    grid_centroids[1, 0] = y_max - raster_profile.pixel_size * 0.5

    centroids_coords = np.dot(grid_centroids, np.mgrid[1:2, :num_columns, :num_rows].T.reshape(-1, 3, 1))
    centroids_coords = centroids_coords.squeeze().T

    if centroids_coords.size < grid_threshold:
        logger.warning(
            f"Centroid size {centroids_coords.size} is too small to calculate density. "
            f"The threshold value is {grid_threshold}. "
            "Check if there’s a data issue or decrease pixel size"
        )
        return raster.RasterData(data=np.array([]), crs=raster_profile.crs, transform=raster_profile.transform)

    if centroids_coords.ndim != 2:
        centroids_coords = centroids_coords.reshape(1, -1)

    tr = neighbors.KDTree(centroids_coords)

    del centroids_coords

    points = np.stack(trajectory_gdf.geometry.apply(lambda x: np.array(x.coords.xy)))

    time = trajectory_gdf.timespan_seconds.values / 3600
    r = maxspeed * time * 1000

    start = tr.query_radius(points[:, :, 0], r=r, return_distance=True)
    end = tr.query_radius(points[:, :, 1], r=r, return_distance=True)

    del tr, points, r

    shape = weibull_pdf.shape
    scale = weibull_pdf.scale

    x = np.arange(0.001, maxspeed, 0.001)
    y = (4 * shape * scale ** (-shape) / np.pi) * np.array(
        [scipy.integrate.quad(_etd, m, maxspeed, args=(shape, scale, m))[0] for m in x]
    )

    raster_ndarray = np.zeros(num_rows * num_columns, dtype=np.float64)

    for k in range(len(start[0])):
        a, b, c = np.intersect1d(start[0][k], end[0][k], return_indices=True)
        speeds = (start[1][k][b] + end[1][k][c]) * 0.001 / time[k]
        i = speeds < maxspeed

        vals = y[np.digitize(speeds[i], x[:-1])] / time[k]

        raster_ndarray[a[i]] += vals / vals.sum() / time[k]

    # Normalize the grid values
    raster_ndarray = raster_ndarray / raster_ndarray.sum()

    ndarray = raster_ndarray.reshape(num_rows, num_columns)

    # write raster_ndarray to GeoTIFF file.
    if output_path:
        # Set the null data values
        raster_ndarray[np.isnan(raster_ndarray) | (raster_ndarray == 0)] = raster_profile.nodata_value

        raster.RasterPy.write(
            ndarray,
            fp=output_path,
            **raster_profile,
        )

    return raster.RasterData(data=ndarray.astype("float32"), crs=raster_profile.crs, transform=raster_profile.transform)


def grid_size_from_geographic_extent(gdf: gpd.GeoDataFrame, scale_factor: int = 10) -> float:
    gdf = gdf.to_crs("EPSG:4326")

    def _haversine_diagonal(bbox: BoundingBox):
        # From https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.haversine_distances.html
        bbox_radians = [math.radians(x) for x in bbox]
        min_point = [bbox_radians[0], bbox_radians[1]]
        max_point = [bbox_radians[2], bbox_radians[3]]
        result = haversine_distances([min_point, max_point])
        result = result * 6371000 / 100  # multiply by Earth radius to get metres
        return result[0][1]

    local_bounds = tuple(gdf.geometry.total_bounds.tolist())
    local_cell_size = int(round(_haversine_diagonal(local_bounds) / scale_factor, 0))

    return local_cell_size
