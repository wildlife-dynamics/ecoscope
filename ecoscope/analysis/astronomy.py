import typing
import warnings
from datetime import datetime, timedelta

import geopandas as gpd  # type: ignore[import-untyped]
import numpy as np
import pandas as pd
import pyproj
import pytz
import shapely
from shapely.geometry.base import BaseGeometry

try:
    import astroplan  # type: ignore[import-untyped]
    import astropy.units as u  # type: ignore[import-untyped]
    from astropy.coordinates import EarthLocation  # type: ignore[import-untyped]
    from astropy.coordinates.erfa_astrom import (  # type: ignore[import-untyped]
        ErfaAstromInterpolator,
        erfa_astrom,
    )
    from astropy.time import Time  # type: ignore[import-untyped]
except ModuleNotFoundError:
    raise ModuleNotFoundError(
        'Missing optional dependencies required by this module. \
         Please run pip install ecoscope["analysis"]'
    )


# Default sample spacing for the astrom interpolator. 1h sits at the plateau
# of speedup-vs-correctness across day/week/month trajectory spans.
DEFAULT_IS_NIGHT_TIME_RESOLUTION = 1 * u.hour


def to_EarthLocation(geometry: gpd.GeoSeries) -> EarthLocation:
    """
    Location on Earth, initialized from geocentric coordinates.

    Parameters
    ----------
    geometry: gpd.GeoSeries
        GeoDataFrame's geometry column

    Returns
    -------
    astropy.coordinates.EarthLocation.
    """
    geometry = geometry.to_crs(4326)
    trans = pyproj.Transformer.from_proj(
        proj_from="+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs",
        proj_to="+proj=geocent +ellps=WGS84 +datum=WGS84 +units=m +no_defs",
    )
    return EarthLocation.from_geocentric(
        *trans.transform(xx=geometry.x, yy=geometry.y, zz=np.zeros(geometry.shape[0])), unit="m"
    )


def is_night(
    geometry: gpd.GeoSeries,
    time: pd.Series,
    time_resolution: u.Quantity = DEFAULT_IS_NIGHT_TIME_RESOLUTION,
) -> pd.Series:
    """
    Classify each (geometry, time) pair as night vs day.

    Parameters
    ----------
    geometry, time: aligned series of locations and timestamps.
    time_resolution: sample spacing for astropy's ErfaAstromInterpolator. Smaller
        values give more accurate results near sunrise/sunset at the cost
        of execution speed; larger values are much faster, introducing sub-degree
        errors in sun altitude. Defaults to 1 hour.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "Geometry is in a geographic CRS.", UserWarning)
        coords = geometry if (geometry.geom_type == "Point").all() else geometry.centroid
        with erfa_astrom.set(ErfaAstromInterpolator(time_resolution)):
            return astroplan.Observer(to_EarthLocation(coords)).is_night(time)


def sun_time(date: datetime, geometry: BaseGeometry) -> pd.Series:
    """
    Sunrise and sunset of the local solar day labelled by `date` at `geometry`.
    Returned timestamps are in UTC, representing the UTC time of the local sunrise/sunset.

    The `geometry` provided is assumed to be in EPSG:4326 (WGS84 lon/lat)
    """
    centroid = geometry.centroid
    offset = timedelta(hours=centroid.x / 15.0)
    local_noon_utc = datetime(date.year, date.month, date.day, 12) - offset
    # Anchoring the search at local solar noon guarantees both events lie on the same local day
    anchor = Time(local_noon_utc, scale="utc")
    observer = astroplan.Observer(location=EarthLocation(lon=centroid.x, lat=centroid.y))
    sunrise = observer.sun_rise_time(anchor, which="previous", n_grid_points=150).to_datetime(timezone=pytz.UTC)
    sunset = observer.sun_set_time(anchor, which="next", n_grid_points=150).to_datetime(timezone=pytz.UTC)
    # astroplan returns a masked 0-d array when it cannot bracket the event within its
    # bounded search window (polar day/night). Coerce to NaT so the day is dropped from
    # the night/day ratio instead of crashing the downstream Timestamp comparisons in
    # calculate_day_fraction with "iteration over a 0-d array".
    if not isinstance(sunrise, datetime):
        sunrise = pd.NaT
    if not isinstance(sunset, datetime):
        sunset = pd.NaT
    return pd.Series({"sunrise": sunrise, "sunset": sunset})


@typing.no_type_check
def calculate_day_night_distance(
    date: datetime, segment_start: datetime, segment_end: datetime, dist_meters: int, daily_summary: pd.DataFrame
) -> None:
    sunrise = daily_summary.loc[date, "sunrise"]
    sunset = daily_summary.loc[date, "sunset"]

    if segment_start < sunset and segment_end > sunset:  # start in day and end in night
        day_percent = (sunset - segment_start) / (segment_end - segment_start)
    elif segment_start < sunrise and segment_end > sunrise:  # start in night and end in day
        day_percent = (segment_end - sunrise) / (segment_end - segment_start)
    elif sunrise < sunset:
        if segment_end < sunrise or segment_start > sunset:  # all night
            day_percent = 0
        elif segment_start >= sunrise and segment_end <= sunset:  # all day
            day_percent = 1
    else:  # sunrise >= sunset
        if segment_end < sunset or segment_start > sunrise:  # all day
            day_percent = 1
        elif segment_start >= sunset and segment_end <= sunrise:  # all night
            day_percent = 0

    daily_summary.loc[date, "day_distance"] += day_percent * dist_meters
    daily_summary.loc[date, "night_distance"] += (1 - day_percent) * dist_meters


def _datetimes_to_epochs(series: pd.Series) -> np.ndarray:
    """Datetime Series -> float array of absolute UTC nanoseconds, with NaT mapped to NaN.

    Comparing/subtracting in absolute UTC lets tz-aware and tz-naive inputs be mixed
    freely (segment times may be tz-aware, sunrise/sunset come back from `sun_time` in
    UTC), and keeps NaN propagating through the arithmetic so unresolved polar days drop out.
    """
    dt = pd.to_datetime(series)
    if getattr(dt.dtype, "tz", None) is not None:
        dt = dt.dt.tz_convert("UTC").dt.tz_localize(None)
    epochs = dt.to_numpy(dtype="datetime64[ns]").view("int64").astype("float64")
    epochs[dt.isna().to_numpy()] = np.nan
    return epochs


def calculate_day_fraction(
    sunrise: pd.Series,
    sunset: pd.Series,
    segment_start: pd.Series,
    segment_end: pd.Series,
) -> np.ndarray:
    """
    Vectorized fraction of each [segment_start, segment_end] interval that falls in
    daylight, given the sunrise/sunset of the local solar day the interval lies in.

    Computed as the overlap of the interval with the daylight window, rather than by
    branching on a single sunrise/sunset crossing. This means an interval that contains
    *both* sunrise and sunset (a full local day) is handled correctly, not only single
    transitions -- callers must first split multi-day segments on local-day boundaries
    (see `get_nightday_ratio`) so exactly one sunrise/sunset pair applies per row.

    Handles both `sunrise < sunset` (normal) and `sunrise >= sunset` (inverted /
    high-latitude) orderings. Times are compared in absolute UTC, so tz-aware and
    tz-naive inputs may be mixed. Rows with NaT sunrise/sunset (polar day/night that
    astroplan could not resolve) yield NaN, which the caller drops.
    """
    start = _datetimes_to_epochs(segment_start)
    end = _datetimes_to_epochs(segment_end)
    rise = _datetimes_to_epochs(sunrise)
    fall = _datetimes_to_epochs(sunset)

    duration = end - start
    # Normal day: daylight is the window [sunrise, sunset]; day fraction is its overlap
    # with the segment. Inverted day: night is the window [sunset, sunrise], and the day
    # fraction is the complement of that overlap.
    day_overlap = np.clip(np.minimum(end, fall) - np.maximum(start, rise), 0.0, None)
    night_overlap = np.clip(np.minimum(end, rise) - np.maximum(start, fall), 0.0, None)

    with np.errstate(invalid="ignore"):
        return np.where(rise < fall, day_overlap / duration, 1.0 - night_overlap / duration)


def _segments_by_local_day(
    local_start: np.ndarray,
    local_end: np.ndarray,
    offset: np.ndarray,
    dist_meters: np.ndarray,
    start_points: gpd.GeoSeries,
) -> pd.DataFrame:
    """Cut a trajectory segment into pieces per local solar day it spans.

    A segment that straddles local midnight -- most often a long gap-bridging segment
    created when a tag stops reporting for hours or days -- is divided at each midnight,
    with its distance apportioned to each piece in proportion to that piece's share of
    the segment's duration. Single-day segments (the overwhelming majority) pass through
    unchanged.

    `local_*` are datetime64[ns] on the local-solar clock (UTC shifted by `offset`).
    Returned `segment_start`/`segment_end` are shifted back to absolute UTC so they line
    up with the UTC sunrise/sunset produced by `sun_time`.
    """
    day = np.timedelta64(1, "D")
    first_day = local_start.astype("datetime64[D]").astype("datetime64[ns]")
    last_day = local_end.astype("datetime64[D]").astype("datetime64[ns]")
    n_days = ((last_day - first_day) / day).astype("int64") + 1

    src = np.repeat(np.arange(len(n_days)), n_days)
    ordinal = np.arange(len(src)) - np.repeat(np.cumsum(n_days) - n_days, n_days)

    seg_start = local_start[src]
    seg_end = local_end[src]
    piece_day = first_day[src] + ordinal * day
    piece_start = np.maximum(seg_start, piece_day)
    piece_end = np.minimum(seg_end, piece_day + day)

    seg_seconds = (seg_end - seg_start) / np.timedelta64(1, "s")
    piece_seconds = (piece_end - piece_start) / np.timedelta64(1, "s")
    piece_dist = dist_meters[src] * (piece_seconds / seg_seconds)

    off = offset[src]
    return pd.DataFrame(
        {
            "local_date": pd.DatetimeIndex(piece_day).date,
            "geometry": list(start_points.values[src]),
            "segment_start": pd.DatetimeIndex(piece_start - off).tz_localize("UTC"),
            "segment_end": pd.DatetimeIndex(piece_end - off).tz_localize("UTC"),
            "dist_meters": piece_dist,
        }
    )


def get_nightday_ratio(gdf: gpd.GeoDataFrame) -> float:
    """Mean of the per-local-day night/day movement ratios.

    Each local solar day contributes ``night_distance / day_distance``, and the returned
    value is the mean of those ratios across days (days with no daytime movement fall out).
    Multi-day segments are first split at local-midnight boundaries (see
    ``_segments_by_local_day``) so a single long gap-bridging segment is apportioned across
    the days it spans instead of being scored entirely against one day's sunrise/sunset.
    Returns NaN if no day has daytime movement.
    """
    start_points = gpd.GeoSeries(
        shapely.get_point(gdf["geometry"].values, 0),
        crs=gdf.crs,
        index=gdf.index,
    ).to_crs(4326)

    # Bin by local solar date to prevent UTC timestamps straddling local night -> day.
    # NOTE: this calculation will skew if tracks cross the -180, 180 boundary
    offset = pd.to_timedelta(start_points.x / 15.0, unit="h")
    local_start = gdf["segment_start"].dt.tz_convert("UTC").dt.tz_localize(None).to_numpy() + offset.to_numpy()
    local_end = gdf["segment_end"].dt.tz_convert("UTC").dt.tz_localize(None).to_numpy() + offset.to_numpy()

    pieces = _segments_by_local_day(
        local_start, local_end, offset.to_numpy(), gdf["dist_meters"].to_numpy(), start_points
    )

    daily_summary = pieces.groupby("local_date").agg(geometry=("geometry", "first"))
    daily_summary[["sunrise", "sunset"]] = daily_summary.apply(lambda x: sun_time(x.name, x.geometry), axis=1)

    day_fraction = calculate_day_fraction(
        sunrise=pieces["local_date"].map(daily_summary["sunrise"]),
        sunset=pieces["local_date"].map(daily_summary["sunset"]),
        segment_start=pieces["segment_start"],
        segment_end=pieces["segment_end"],
    )

    dist = pieces["dist_meters"].to_numpy()
    day_dist = pd.Series(day_fraction * dist, index=pieces.index).groupby(pieces["local_date"]).sum()
    night_dist = pd.Series((1.0 - day_fraction) * dist, index=pieces.index).groupby(pieces["local_date"]).sum()

    night_day_ratio = night_dist / day_dist
    return night_day_ratio.replace([np.inf, -np.inf], np.nan).dropna().mean()
