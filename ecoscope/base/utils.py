from datetime import datetime
import geopandas as gpd  # type: ignore[import-untyped]
import numpy as np
import pandas as pd
from pandas.core.groupby.generic import DataFrameGroupBy
from shapely.geometry import box
from shapely.geometry.base import BaseGeometry
from typing import Any, Literal, Tuple

BoundingBox = Tuple[float, float, float, float]


def create_meshgrid(
    aoi: BaseGeometry,
    in_crs: Any,
    out_crs: Any = "EPSG:3857",
    xlen: int = 1000,
    ylen: int = 1000,
    return_intersecting_only: bool = True,
    align_to_existing: gpd.GeoSeries | gpd.GeoDataFrame | None = None,
) -> gpd.GeoSeries:
    """Create a grid covering `aoi`.

    Parameters
    ----------
    aoi : shapely.geometry.base.BaseGeometry
        The area of interest.
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
    aoi = a.to_crs(out_crs)[0]
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

        align_to_existing = align_to_existing.to_crs(out_crs)

        total_existing_bounds = align_to_existing.total_bounds

        x1 += total_existing_bounds[0] % xlen - xlen
        y1 += total_existing_bounds[1] % ylen - ylen

    boxes = [box(x, y, x + xlen, y + ylen) for x in np.arange(x1, x2, xlen) for y in np.arange(y1, y2, ylen)]

    gs = gpd.GeoSeries(boxes, crs=out_crs)
    if return_intersecting_only:
        gs = gs[gs.intersects(aoi)]

    return gs


def groupby_intervals(df: pd.DataFrame, col: str, intervals: pd.IntervalIndex) -> DataFrameGroupBy:
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

    if not df[col].is_monotonic_increasing:
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


def create_interval_index(
    start: datetime | pd.Timestamp,
    intervals: int,
    freq: str | pd.Timedelta | pd.DateOffset,
    overlap: pd.Timedelta | pd.DateOffset = pd.Timedelta(0),
    closed: Literal["left", "right", "both", "neither"] = "right",
    round_down_to_freq: bool = False,
) -> pd.IntervalIndex:
    """
    Parameters
    ----------
    start : Datetime or pd.Timestamp
        Left bound for creating IntervalIndex
    intervals : int, optional
        Number of intervals to create
    freq : Timedelta or DateOffset
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

    freq_offset: pd.tseries.offsets.BaseOffset = pd.tseries.frequencies.to_offset(freq)  # type: ignore[arg-type]
    overlap_offset: pd.tseries.offsets.BaseOffset = pd.tseries.frequencies.to_offset(overlap)  # type: ignore[arg-type]

    if round_down_to_freq:
        start = start.round(freq_offset)  # type: ignore[arg-type, union-attr]

    left = [start]
    for i in range(1, intervals):
        start = start + freq_offset
        left.append(start - i * overlap_offset)
    left_index = pd.DatetimeIndex(left)

    return pd.IntervalIndex.from_arrays(left=left_index, right=left_index + freq_offset, closed=closed)


class ModisBegin(pd._libs.tslibs.offsets.SingleConstructorOffset):
    """
    Primitive DateOffset to support MODIS period start times.
    """

    def apply(self, other: pd.Timestamp) -> pd.Timestamp:
        assert other.tz is not None, "Timestamp must be timezone-aware"
        other = other.astimezone("UTC").round("d")  # type: ignore[arg-type]
        for i in range(self.n):
            other = min(other + pd.offsets.YearBegin(), other + pd.DateOffset(days=(16 - (other.dayofyear - 1) % 16)))
        return other


def create_modis_interval_index(
    start: str | datetime | pd.Timestamp,
    intervals: int,
    overlap: pd.Timedelta | pd.DateOffset = pd.Timedelta(0),
    closed: Literal["left", "right", "both", "neither"] = "right",
) -> pd.IntervalIndex:
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

    modis = ModisBegin()
    start = modis.apply(pd.Timestamp(start))

    left = [start]
    for i in range(1, intervals):
        start = modis.apply(start)
        left.append(start - i * overlap)
    left_index = pd.DatetimeIndex(left)

    return pd.IntervalIndex.from_arrays(left=left_index, right=left_index + pd.Timedelta(days=16), closed=closed)


def add_val_index(df: pd.DataFrame, index_name: str, val: str) -> pd.DataFrame:
    """
    Parameters
    ----------
    df : pd.DataFrame
        Dataframe to create index on
    index_name : str
        Name of index
    val : str
        Current column to rename and set as index

    Returns
    -------
    pd.DataFrame

    """
    if val in df.columns:
        return df.rename(columns={val: index_name}).set_index(index_name, append=True)
    else:
        df[index_name] = val
        df = df.set_index(index_name, append=True)
    return df


def add_temporal_index(df: pd.DataFrame, index_name: str, time_col: str, directive: str) -> pd.DataFrame:
    """
    Parameters
    ----------
    df : pd.DataFrame
        Dataframe to create index on
    index_name : str
        Name of temporal index
    time_col : str
        Name of the time column
    directive : str
        Time format string

    Returns
    -------
    pd.DataFrame

    """
    if directive and time_col:
        df[index_name] = df[time_col].dt.strftime(directive)
        return df.set_index(index_name, append=True)
    else:
        return df


def hex_to_rgba(input: str) -> tuple:
    if not input:
        raise ValueError("Input cannot be empty")
    hex = input.strip("#")

    if len(hex) != 6 and len(hex) != 8:
        raise ValueError("Invalid hex length, must be 6 or 8")

    # Max alpha if none provided
    if len(hex) == 6:
        hex = f"{hex}FF"

    try:
        return tuple(int(hex[i : i + 2], 16) for i in (0, 2, 4, 6))
    except ValueError:
        raise ValueError(f"Invalid hex string, {input}")


def color_tuple_to_css(color: Tuple[int, int, int, int]) -> str:
    # eg [255,0,120,255] converts to 'rgba(255,0,120,1)'
    return f"rgba({color[0]}, {color[1]}, {color[2]}, {color[3]/255})"
