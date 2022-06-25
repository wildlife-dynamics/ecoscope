import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import box


def create_meshgrid(
    aoi,
    in_crs,
    out_crs,
    xlen=1000,
    ylen=1000,
    return_intersecting_only=True,
    align_to_existing=None,
):
    """Create a grid covering `aoi`.

    Parameters
    ----------
    aoi : shapely.geometry.base.BaseGeometry
        The area of interest. Should be in a UTM CRS.
    in_crs : value
        Coordinate Reference System of input `aoi`. Can be anything accepted by `pyproj.CRS.from_user_input()`.
        Geometry is automatically converted to UTM CRS as an intermediate for computation.
    out_crs : value
        Coordinate Reference System of output `gs`. Can be anything accepted by `pyproj.CRS.from_user_input()`.
        Geometry is automatically converted to UTM CRS as an intermediate for computation.
    xlen : int, optional
        The width of a grid cell in meters.
    ylen : int, optional
        The height of a grid cell in meters.
    return_intersecting_only : bool, optional
        Whether to return only grid cells intersecting with the aoi.
    align_to_existing : geopandas.GeoSeries or geopandas.GeoDataFrame, optional
        If provided, attempts to align created grid to start of existing grid. Requires a CRS and valid geometry.

    Returns
    -------
    gs : geopandas.GeoSeries
        Grid of boxes. CRS is converted to `out_crs`.
    """

    a = gpd.array.from_shapely([aoi], crs=in_crs)
    int_crs = a.estimate_utm_crs()
    aoi = a.to_crs(int_crs)[0]
    del a

    bounds = aoi.bounds

    x1 = bounds[0] - bounds[0] % xlen
    y1 = bounds[1] - bounds[1] % ylen
    x2 = bounds[2]
    y2 = bounds[3]

    if align_to_existing is not None:
        align_to_existing = align_to_existing.geometry  # Treat GeoDataFrame as GeoSeries
        assert not align_to_existing.isna().any()
        assert not align_to_existing.is_empty.any()

        align_to_existing = align_to_existing.to_crs(int_crs)

        total_existing_bounds = align_to_existing.total_bounds

        x1 += total_existing_bounds[0] % xlen - xlen
        y1 += total_existing_bounds[1] % ylen - ylen

    boxes = [box(x, y, x + xlen, y + ylen) for x in np.arange(x1, x2, xlen) for y in np.arange(y1, y2, ylen)]

    gs = gpd.GeoSeries(boxes, crs=int_crs)
    if return_intersecting_only:
        gs = gs[gs.intersects(aoi)]

    return gs.to_crs(out_crs)


class cachedproperty:  # noqa
    """
    The ``cachedproperty`` is used similar to :class:`property`, except
    that the wrapped method is only called once. This is commonly used
    to implement lazy attributes.
    """

    def __init__(self, func):
        self.func = func

    def __doc__(self):
        return getattr(self.func, "__doc__")

    def __isabstractmethod__(self):
        return getattr(self.func, "__isabstractmethod__", False)

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        value = obj.__dict__[self.func.__name__] = self.func(obj)
        return value

    def __repr__(self):
        cls = self.__class__.__name__
        return f"<{cls} func={self.func}>"


def groupby_intervals(df, col, intervals):
    """
    Parameters
    ----------
    df : pd.DataFrame
        Data to group
    col : str
        Name of column to group on
    intervals : pd.IntervalIndex
        Intervals to group on

    Returns
    -------
    pd.core.groupby.DataFrameGroupBy

    """

    if not df[col].is_monotonic:
        df = df.sort_values(col)

    return pd.concat(
        {
            interval: df.iloc[left_i:right_i]
            for interval, left_i, right_i in zip(
                intervals,
                df[col].searchsorted(intervals.left, side="left" if intervals.closed in ("left", "both") else "right"),
                df[col].searchsorted(
                    intervals.right, side="right" if intervals.closed in ("right", "both") else "left"
                ),
            )
        }
    ).groupby(level=0)


def create_interval_index(start, intervals, freq, overlap=pd.Timedelta(0), closed="right", round_down_to_freq=False):
    """
    Parameters
    ----------
    start : str or datetime-like
        Left bound for creating IntervalIndex
    intervals : int, optional
        Number of intervals to create
    freq : str, Timedelta or DateOffset
        Length of each interval
    overlap : Timedelta or DateOffset, optional
        Length of overlap between intervals
    closed : {"left", "right", "both", "neither"}, default "right"
        Whether the intervals are closed on the left-side, right-side, both or neither
    round_down_to_freq : bool
        Start will be rounded down to freq

    Returns
    -------
    interval_index : pd.IntervalIndex

    """

    freq = pd.tseries.frequencies.to_offset(freq)
    overlap = pd.tseries.frequencies.to_offset(overlap)
    if round_down_to_freq:
        start = start.round(freq)

    left = [start]
    for i in range(1, intervals):
        start = start + freq
        left.append(start - i * overlap)
    left = pd.DatetimeIndex(left)

    return pd.IntervalIndex.from_arrays(left=left, right=left + freq, closed=closed)


class ModisBegin(pd._libs.tslibs.offsets.SingleConstructorOffset):
    """
    Primitive DateOffset to support MODIS period start times.
    """

    def apply(self, other: pd.Timestamp) -> pd.Timestamp:
        assert other.tz is not None, "Timestamp must be timezone-aware"
        other = other.astimezone("UTC").round("d")
        for i in range(self.n):
            other = min(other + pd.offsets.YearBegin(), other + pd.DateOffset(days=(16 - (other.dayofyear - 1) % 16)))
        return other


def create_modis_interval_index(start, intervals, overlap=pd.Timedelta(0), closed="right"):
    """
    Parameters
    ----------
    start : str or datetime-like
        Left bound for creating IntervalIndex. Will be rounded up to next start of MODIS period
    intervals : int, optional
        Number of intervals to create
    overlap : Timedelta or DateOffset, optional
        Length of overlap between intervals
    closed : {"left", "right", "both", "neither"}, default "right"
        Whether the intervals are closed on the left-side, right-side, both or neither

    Returns
    -------
    interval_index : pd.IntervalIndex

    """

    start = start + ModisBegin()

    left = [start]
    for i in range(1, intervals):
        start = start + ModisBegin()
        left.append(start - i * overlap)
    left = pd.DatetimeIndex(left)

    return pd.IntervalIndex.from_arrays(left=left, right=left + pd.Timedelta(days=16), closed=closed)
