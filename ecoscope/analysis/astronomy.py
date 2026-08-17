import logging
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


logger = logging.getLogger(__name__)


# Default sample spacing for the astrom interpolator. 1h sits at the plateau
# of speedup-vs-correctness across day/week/month trajectory spans.
DEFAULT_IS_NIGHT_TIME_RESOLUTION = 1 * u.hour

# Sun-altitude threshold defining "night" for the night/day ratio. -6 deg is civil
# twilight: below it there is meaningful darkness. Using the true horizon (0 deg)
# instead would count as "night" hours that never actually get dark at high latitude
# (e.g. 82 days a year at lat 60, 86 at 66.5).
DEFAULT_NIGHT_HORIZON = -6 * u.deg

# A phase (night or day) of a solar unit must have at least this many *observed* hours
# for the unit to yield a usable night-vs-day speed comparison. Thinner coverage can't be
# trusted and would divide a small distance by a near-zero time, so the unit is set aside.
DEFAULT_MIN_PHASE_HOURS = 1.0

# Trajectory segments longer than this are dropped from the night/day ratio rather than
# apportioned across the phases they span: a long gap-bridging chord (a tag that stopped
# reporting) has no real within-segment speed, so splitting it by elapsed time would just
# assume constant speed -- the very thing the ratio is trying to measure.
DEFAULT_MAX_SEGMENT_GAP_HOURS = 6.0

# Grid resolution for astroplan's rise/set search over the ~24h window. 720 points is
# ~2-minute resolution -- fine enough to bracket a short high-latitude night, where the
# 150-point default (~10 min) is too coarse.
DEFAULT_SUN_TIME_N_GRID_POINTS = 720


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


def night_window(
    date: datetime,
    geometry: BaseGeometry,
    horizon: u.Quantity = DEFAULT_NIGHT_HORIZON,
    n_grid_points: int = DEFAULT_SUN_TIME_N_GRID_POINTS,
) -> pd.Series:
    """
    The night window -- civil dusk to the following civil dawn -- bracketing the local
    midnight of the solar unit labelled by `date` at `geometry`.

    A solar unit runs from local noon to local noon, so its one night sits contiguously in
    the middle around local midnight. `dusk` is the last time the sun drops below `horizon`
    before that midnight, `dawn` the first time it climbs back above `horizon` after it.
    `horizon` defaults to -6 deg (civil twilight); everything outside [dusk, dawn] within the
    unit is day. Returned timestamps are in UTC.

    If the sun never crosses `horizon` (a high-latitude "bright night", or polar day/night),
    astroplan returns a masked 0-d array; that event is coerced to NaT, meaning "no qualifying
    darkness" -- the unit then has no night window and is dropped by `get_nightday_ratio`.

    The `geometry` provided is assumed to be in EPSG:4326 (WGS84 lon/lat).
    """
    centroid = geometry.centroid
    offset = timedelta(hours=centroid.x / 15.0)
    local_midnight_utc = datetime(date.year, date.month, date.day) - offset
    # Anchoring the search at local midnight puts dusk (previous) and dawn (next) on either
    # side of the same night, so [dusk, dawn] is a single contiguous window.
    anchor = Time(local_midnight_utc, scale="utc")
    observer = astroplan.Observer(location=EarthLocation(lon=centroid.x, lat=centroid.y))
    with warnings.catch_warnings():
        # A sun that never crosses `horizon` is an expected "no qualifying darkness" outcome we
        # handle below as NaT, not a condition worth warning about once per polar unit.
        warnings.filterwarnings("ignore", "Target with index .* does not cross horizon")
        dusk = observer.sun_set_time(
            anchor, which="previous", horizon=horizon, n_grid_points=n_grid_points
        ).to_datetime(timezone=pytz.UTC)
        dawn = observer.sun_rise_time(anchor, which="next", horizon=horizon, n_grid_points=n_grid_points).to_datetime(
            timezone=pytz.UTC
        )
    # Coerce astroplan's masked 0-d array (no crossing found) to NaT so the unit drops out of
    # the ratio instead of crashing the downstream Timestamp comparisons in
    # calculate_night_fraction with "iteration over a 0-d array".
    if not isinstance(dusk, datetime):
        dusk = pd.NaT
    if not isinstance(dawn, datetime):
        dawn = pd.NaT
    return pd.Series({"dusk": dusk, "dawn": dawn})


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


def calculate_night_fraction(
    dusk: pd.Series,
    dawn: pd.Series,
    segment_start: pd.Series,
    segment_end: pd.Series,
) -> np.ndarray:
    """
    Vectorized fraction of each [segment_start, segment_end] interval that falls in
    darkness, given the [dusk, dawn] night window of the solar unit the interval lies in.

    The night window is a single contiguous interval bracketing local midnight (see
    `night_window`), so the night fraction is simply its overlap with the segment divided by
    the segment's duration -- no branching on day ordering is needed. Callers must first
    split segments on local-noon boundaries (see `get_nightday_ratio`) so exactly one night
    window applies per row.

    Times are compared in absolute UTC, so tz-aware and tz-naive inputs may be mixed. Rows
    with NaT dusk/dawn (no qualifying darkness that astroplan could resolve) yield NaN, which
    the caller drops.
    """
    start = _datetimes_to_epochs(segment_start)
    end = _datetimes_to_epochs(segment_end)
    fall = _datetimes_to_epochs(dusk)
    rise = _datetimes_to_epochs(dawn)

    duration = end - start
    night_overlap = np.clip(np.minimum(end, rise) - np.maximum(start, fall), 0.0, None)

    with np.errstate(invalid="ignore"):
        return night_overlap / duration


def _segments_by_solar_day(
    local_start: np.ndarray,
    local_end: np.ndarray,
    offset: np.ndarray,
    dist_meters: np.ndarray,
    start_points: gpd.GeoSeries,
) -> pd.DataFrame:
    """Cut a trajectory segment into pieces per local solar unit (noon to noon) it spans.

    A unit runs from one local solar noon to the next, so its single night sits whole in the
    middle. A segment that straddles a local-noon boundary is divided at it, with its distance
    and time apportioned to each piece in proportion to that piece's share of the segment's
    duration. Single-unit segments (the overwhelming majority) pass through unchanged.

    `local_*` are datetime64[ns] on the local-solar clock (UTC shifted by `offset`). Flooring
    is done on the clock shifted a further 12h so day boundaries land on local noon; the
    returned `local_date` labels each unit by the date of the local midnight at its centre,
    which is what `night_window` expects. Returned `segment_start`/`segment_end` are converted
    back to absolute UTC so they line up with the UTC dusk/dawn produced by `night_window`.
    """
    day = np.timedelta64(1, "D")
    twelve_h = np.timedelta64(12, "h")
    # Shift by 12h before flooring so unit boundaries fall on local *solar* noon rather than
    # midnight; the floored date then equals the date of the local midnight at the unit's centre.
    # The 12h is a clock offset -- we treat mean solar time as UTC + lon/15h, so solar noon is
    # 12:00 on that clock at every latitude; it is the night's *length* that varies with latitude,
    # not this noon-to-midnight offset. Anchoring the boundary at solar noon (the point farthest
    # from the night) keeps even a long high-latitude night whole within one unit, and the
    # equation-of-time error (~16 min) in the mean-solar approximation lands in daytime, never
    # near a dusk/dawn edge where it could misclassify movement.
    shifted_start = local_start + twelve_h
    shifted_end = local_end + twelve_h
    first_day = shifted_start.astype("datetime64[D]").astype("datetime64[ns]")
    last_day = shifted_end.astype("datetime64[D]").astype("datetime64[ns]")
    n_days = ((last_day - first_day) / day).astype("int64") + 1

    src = np.repeat(np.arange(len(n_days)), n_days)
    ordinal = np.arange(len(src)) - np.repeat(np.cumsum(n_days) - n_days, n_days)

    seg_start = shifted_start[src]
    seg_end = shifted_end[src]
    piece_day = first_day[src] + ordinal * day
    piece_start = np.maximum(seg_start, piece_day)
    piece_end = np.minimum(seg_end, piece_day + day)

    seg_seconds = (seg_end - seg_start) / np.timedelta64(1, "s")
    piece_seconds = (piece_end - piece_start) / np.timedelta64(1, "s")
    piece_dist = dist_meters[src] * (piece_seconds / seg_seconds)

    # Undo the 12h noon-shift and the longitude offset to land back on absolute UTC.
    off = offset[src]
    return pd.DataFrame(
        {
            "local_date": pd.DatetimeIndex(piece_day).date,
            "geometry": list(start_points.values[src]),
            "segment_start": pd.DatetimeIndex(piece_start - twelve_h - off).tz_localize("UTC"),
            "segment_end": pd.DatetimeIndex(piece_end - twelve_h - off).tz_localize("UTC"),
            "dist_meters": piece_dist,
            "obs_seconds": piece_seconds,
        }
    )


class NightDayRatio(typing.NamedTuple):
    """Result of :func:`get_nightday_ratio`.

    ``ratio`` is the nocturnality ratio (1.0 balanced, >1 nocturnal, <1 diurnal; NaN if no
    unit was measurable, inf if fully nocturnal). ``n_days`` is the number of solar units that
    actually contributed -- the ratio can't be interpreted or compared without it, since at
    high latitude it may rest on only the shoulder-season days that had a qualifying night.
    """

    ratio: float
    n_days: int


def get_nightday_ratio(
    gdf: gpd.GeoDataFrame,
    *,
    night_horizon: u.Quantity = DEFAULT_NIGHT_HORIZON,
    min_phase_hours: float = DEFAULT_MIN_PHASE_HOURS,
    max_segment_gap_hours: float = DEFAULT_MAX_SEGMENT_GAP_HOURS,
    n_grid_points: int = DEFAULT_SUN_TIME_N_GRID_POINTS,
) -> NightDayRatio:
    """Nocturnality ratio: how much faster an animal moves in darkness vs daylight.

    Each local solar unit (noon to noon, one whole night in the middle) is scored by comparing
    movement *speed* -- distance per observed hour -- in darkness against daylight, rather than
    raw distance. Speed removes the night-length bias that makes a cathemeral animal look
    diurnal in Arctic summer and nocturnal in winter simply because the night is short/long.

    Per unit::

        night_speed = night_distance / night_hours     # m per observed hour of darkness
        day_speed   = day_distance   / day_hours        # m per observed hour of daylight
        share       = night_speed / (night_speed + day_speed)   # in [0, 1], 0.5 == balanced

    where "night" is time below ``night_horizon`` (default -6 deg, civil twilight; see
    ``night_window``) and *hours* are hours actually observed, not the length of the
    astronomical window. The per-unit shares are averaged weighted by each unit's coverage
    (total observed hours), then mapped back to the familiar ratio scale via
    ``mean_share / (1 - mean_share)`` so 1.0 is balanced, >1 nocturnal, <1 diurnal.

    A unit is set aside (and excluded from ``n_days``) unless *both* phases have more than
    ``min_phase_hours`` of observed data: without coverage of both, the speed comparison is
    meaningless, and a unit with no qualifying darkness (high-latitude bright night / polar day)
    simply has zero night hours and drops out -- so there is no polar special case.

    Segments longer than ``max_segment_gap_hours`` are dropped rather than apportioned: a long
    gap-bridging chord has no meaningful within-segment speed, and splitting it by elapsed time
    would assume constant speed -- the very thing being measured. Shorter segments straddling a
    noon or dusk/dawn boundary are split (see ``_segments_by_solar_day``).

    Returns a :class:`NightDayRatio` ``(ratio, n_days)``. ``ratio`` is NaN when no unit is
    measurable and inf for a fully nocturnal track (every measurable unit had zero daytime
    movement).

    NOTE: the ratio is sensitive to fix rate -- it is not comparable across tracks with
    different duty cycles (coarsening the fix interval 16x drags a 1.53 down to 1.10), because
    a straight-line segment understates the true path more the longer it is. It also skews if a
    track crosses the -180/180 meridian.
    """
    # Drop long gap-bridging segments before any splitting: their straight-line speed is not a
    # real speed, so they must not enter the night-vs-day speed comparison at all (#7).
    timespan_hours = (gdf["segment_end"] - gdf["segment_start"]).dt.total_seconds() / 3600.0
    keep = timespan_hours <= max_segment_gap_hours
    n_dropped = int((~keep).sum())
    if n_dropped:
        total_dist = gdf["dist_meters"].sum()
        dropped_share = gdf.loc[~keep, "dist_meters"].sum() / total_dist if total_dist else 0.0
        logger.info(
            "get_nightday_ratio dropped %d/%d segments longer than %.1fh (%.2f%% of distance)",
            n_dropped,
            len(gdf),
            max_segment_gap_hours,
            100.0 * dropped_share,
        )
    gdf = gdf.loc[keep]
    if gdf.empty:
        return NightDayRatio(np.nan, 0)

    start_points = gpd.GeoSeries(
        shapely.get_point(gdf["geometry"].values, 0),
        crs=gdf.crs,
        index=gdf.index,
    ).to_crs(4326)

    # Work on a local-solar clock (UTC shifted by longitude) so noon-to-noon units and their
    # night windows are anchored to the animal's own solar time regardless of input timezone.
    offset = pd.to_timedelta(start_points.x / 15.0, unit="h")
    local_start = gdf["segment_start"].dt.tz_convert("UTC").dt.tz_localize(None).to_numpy() + offset.to_numpy()
    local_end = gdf["segment_end"].dt.tz_convert("UTC").dt.tz_localize(None).to_numpy() + offset.to_numpy()

    pieces = _segments_by_solar_day(
        local_start, local_end, offset.to_numpy(), gdf["dist_meters"].to_numpy(), start_points
    )

    unit_summary = pieces.groupby("local_date").agg(geometry=("geometry", "first"))
    unit_summary[["dusk", "dawn"]] = unit_summary.apply(
        lambda x: night_window(x.name, x.geometry, horizon=night_horizon, n_grid_points=n_grid_points), axis=1
    )

    night_fraction = calculate_night_fraction(
        dusk=pieces["local_date"].map(unit_summary["dusk"]),
        dawn=pieces["local_date"].map(unit_summary["dawn"]),
        segment_start=pieces["segment_start"],
        segment_end=pieces["segment_end"],
    )

    # Split both distance and observed time of every piece into its night and day parts, then
    # total per unit. Units with no qualifying darkness have all-NaN fractions -> zero night
    # hours -> gated out below.
    dist = pieces["dist_meters"].to_numpy()
    hours = pieces["obs_seconds"].to_numpy() / 3600.0
    by_unit = pieces["local_date"]
    night_dist = pd.Series(night_fraction * dist, index=pieces.index).groupby(by_unit).sum()
    day_dist = pd.Series((1.0 - night_fraction) * dist, index=pieces.index).groupby(by_unit).sum()
    night_hours = pd.Series(night_fraction * hours, index=pieces.index).groupby(by_unit).sum()
    day_hours = pd.Series((1.0 - night_fraction) * hours, index=pieces.index).groupby(by_unit).sum()

    # Keep only units with enough observed data in *both* phases to compare speeds (#4).
    gate = (day_hours > min_phase_hours) & (night_hours > min_phase_hours)
    night_speed = night_dist[gate] / night_hours[gate]
    day_speed = day_dist[gate] / day_hours[gate]

    total_speed = night_speed + day_speed
    share = night_speed / total_speed
    weight = night_hours[gate] + day_hours[gate]
    # A gated unit where the animal was observed but stationary in both phases (total_speed 0)
    # says nothing about preference; drop those so they neither skew nor NaN the average.
    valid = total_speed > 0
    n_days = int(valid.sum())
    if n_days == 0:
        return NightDayRatio(np.nan, 0)

    # Weight by coverage so a well-observed unit counts more than a thin sliver of a day (#5).
    mean_share = float(np.average(share[valid], weights=weight[valid]))
    if mean_share >= 1.0:
        return NightDayRatio(np.inf, n_days)
    return NightDayRatio(mean_share / (1.0 - mean_share), n_days)
