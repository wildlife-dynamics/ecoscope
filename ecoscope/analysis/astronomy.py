import typing
import warnings
from datetime import datetime, timedelta

import geopandas as gpd  # type: ignore[import-untyped]
import numpy as np
import pandas as pd
import pyproj
import pytz
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
    midnight = Time(
        datetime(date.year, date.month, date.day) + timedelta(seconds=1), scale="utc"
    )  # add 1 second shift to avoid leap_second_strict warning
    observer = astroplan.Observer(location=EarthLocation(lon=geometry.centroid.x, lat=geometry.centroid.y))
    sunrise = observer.sun_rise_time(midnight, which="next", n_grid_points=150).to_datetime(timezone=pytz.UTC)
    sunset = observer.sun_set_time(midnight, which="next", n_grid_points=150).to_datetime(timezone=pytz.UTC)
    # astroplan returns a masked 0-d array when it cannot bracket the event within its
    # bounded forward search window (e.g. mid-latitude days where the "next" sunrise/sunset
    # falls at the window edge, or polar day/night). Coerce to NaT so the day is dropped from
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


def calculate_day_fraction(
    sunrise: pd.Series,
    sunset: pd.Series,
    segment_start: pd.Series,
    segment_end: pd.Series,
) -> np.ndarray:
    """
    Vectorized fraction of each [segment_start, segment_end] interval that falls
    in daylight, given the corresponding sunrise/sunset per row.

    Handles both `sunrise < sunset` (normal) and `sunrise >= sunset` (inverted /
    high-latitude) day orderings. Boundary samples (segment touching sunrise or
    sunset exactly) are assigned to the same side as the adjacent open interval,
    so np.select never falls through to NaN for valid `segment_start < segment_end`.
    """
    duration = segment_end - segment_start

    sr_lt_ss = sunrise < sunset
    is_day_to_night = (segment_start <= sunset) & (segment_end > sunset)
    is_night_to_day = (segment_start <= sunrise) & (segment_end > sunrise)
    is_all_night_normal = sr_lt_ss & ((segment_end <= sunrise) | (segment_start >= sunset))
    is_all_day_normal = sr_lt_ss & (segment_start >= sunrise) & (segment_end <= sunset)
    is_all_day_inverted = (~sr_lt_ss) & ((segment_end <= sunset) | (segment_start >= sunrise))
    is_all_night_inverted = (~sr_lt_ss) & (segment_start >= sunset) & (segment_end <= sunrise)

    return np.select(
        [
            is_day_to_night,
            is_night_to_day,
            is_all_night_normal,
            is_all_day_normal,
            is_all_day_inverted,
            is_all_night_inverted,
        ],
        [
            (sunset - segment_start) / duration,
            (segment_end - sunrise) / duration,
            0.0,
            1.0,
            1.0,
            0.0,
        ],
        default=np.nan,
    )


def get_nightday_ratio(gdf: gpd.GeoDataFrame) -> float:
    # Ensure UTC here so each date lines up with the UTC midnight reference used by sun_time
    gdf["date"] = gdf["segment_start"].dt.tz_convert("UTC").dt.date

    daily_summary = gdf.groupby("date").first()["geometry"].reset_index()
    daily_summary[["sunrise", "sunset"]] = daily_summary.apply(lambda x: sun_time(x.date, x.geometry), axis=1)
    daily_summary = daily_summary.set_index("date")

    day_fraction = calculate_day_fraction(
        sunrise=gdf["date"].map(daily_summary["sunrise"]),
        sunset=gdf["date"].map(daily_summary["sunset"]),
        segment_start=gdf["segment_start"],
        segment_end=gdf["segment_end"],
    )

    day_dist = day_fraction * gdf["dist_meters"]
    night_dist = (1 - day_fraction) * gdf["dist_meters"]

    daily_summary["day_distance"] = day_dist.groupby(gdf["date"]).sum().reindex(daily_summary.index, fill_value=0.0)
    daily_summary["night_distance"] = night_dist.groupby(gdf["date"]).sum().reindex(daily_summary.index, fill_value=0.0)

    daily_summary["night_day_ratio"] = daily_summary["night_distance"] / daily_summary["day_distance"]
    mean_night_day_ratio = daily_summary["night_day_ratio"].replace([np.inf, -np.inf], np.nan).dropna().mean()
    return mean_night_day_ratio
