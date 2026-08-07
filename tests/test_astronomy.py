from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pyproj
import pytest
from shapely.geometry import Point

from ecoscope import Trajectory
from ecoscope.analysis import astronomy
from tests.conftest import ARCTIC, EQUATOR, Segment, build_segments

# Normal day: sunrise 06:00, sunset 18:00
SUNRISE = datetime(2024, 1, 1, 6, 0)
SUNSET = datetime(2024, 1, 1, 18, 0)
# Inverted day (polar / high-latitude): sunset 06:00, sunrise 18:00
INVERTED_SUNRISE = datetime(2024, 1, 1, 18, 0)
INVERTED_SUNSET = datetime(2024, 1, 1, 6, 0)


def test_to_EarthLocation(movebank_relocations):
    geometry = movebank_relocations.gdf["geometry"]
    test_point = geometry.iloc[0]

    transformed = astronomy.to_EarthLocation(geometry)

    assert len(geometry) == len(transformed)

    transform = pyproj.Transformer.from_proj(
        proj_from="+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs",
        proj_to="+proj=geocent +ellps=WGS84 +datum=WGS84 +units=m +no_defs",
    )

    # Check the projected values in the returned EarthLocation are what we expect
    test_val = transform.transform(xx=test_point.x, yy=test_point.y, zz=0)
    assert test_val[0] == transformed[0].x.value
    assert test_val[1] == transformed[0].y.value
    assert test_val[2] == transformed[0].z.value


def test_is_night(movebank_relocations):
    subset = movebank_relocations.gdf.iloc[12:15].copy()

    subset["is_night"] = astronomy.is_night(subset.geometry, subset.fixtime)

    assert subset["is_night"].values.tolist() == [True, True, False]


@pytest.mark.parametrize(
    "timezone",
    [
        timezone(timedelta(hours=10)),
        timezone.utc,
        timezone(timedelta(hours=-6)),
    ],
)
def test_nightday_ratio_salif_habiba(movebank_relocations, timezone):
    # movebank_relocations is subsampled to keep execution speed low.
    # Expected (mean per-day night fraction mapped to ratio scale) for the full data are:
    # Habiba=0.4420952421286314, Salif Keita=1.4269356538351035.
    movebank_relocations.gdf = movebank_relocations.gdf.groupby("groupby_col", group_keys=False).head(100)

    trajectory = Trajectory.from_relocations(movebank_relocations)
    expected = pd.Series(
        [0.3846537588901658, 1.7533910921716953],
        index=pd.Index(["Habiba", "Salif Keita"], name="groupby_col"),
    )
    # test against a handful of timezone to ensure this calculation is agnotisc of input timezone
    trajectory.gdf["segment_start"] = trajectory.gdf["segment_start"].dt.tz_convert(timezone).dt.as_unit("ns")
    pd.testing.assert_series_equal(
        trajectory.gdf.groupby("groupby_col")[trajectory.gdf.columns].apply(
            astronomy.get_nightday_ratio, include_groups=False
        ),
        expected,
    )


@pytest.mark.parametrize("lon", [0.0, 60.0, 120.0, -60.0, -120.0, 23.0, -77.0])
def test_nightday_ratio_synthetic_baseline(lon):
    # Two clear-night + two clear-day 1h segments -> ratio 1.0 regardless of longitude.
    # For non-zero longitudes the segments straddle two UTC dates, so this also
    # ensures we're agnostic of UTC date boundaries.
    utc_midnight = pd.Timestamp("2024-03-20", tz="UTC")
    tz_offset = pd.Timedelta(hours=lon / 15.0)
    start = Point(lon, 0.0)
    segments = [
        Segment(
            start,
            utc_midnight + pd.Timedelta(hours=h) - tz_offset,
            utc_midnight + pd.Timedelta(hours=h + 1) - tz_offset,
        )
        for h in [2, 12, 13, 22]  # two clearly-night, two clearly-day
    ]
    assert astronomy.get_nightday_ratio(build_segments(segments)) == pytest.approx(1.0, rel=1e-6)


@pytest.mark.parametrize("lon", [150.0, -150.0, 179.0])
def test_nightday_ratio_asymmetric_longitude_no_night_inflation(lon):
    # Regression for mismatches in UTC-vs-local solar-time: at far east/west longitudes,
    # local daytime falls at UTC times that "look like" night. An asymmetric 3:1 day:night
    # distance pins the classification direction -- night/day must be ~0.333, not the ~3.0
    # a UTC-confused classifier would give.
    tz_offset = pd.Timedelta(hours=lon / 15.0)
    start = Point(lon, 0.0)
    day = pd.Timestamp("2024-03-20 10:00") - tz_offset  # local day
    night = pd.Timestamp("2024-03-20 22:00") - tz_offset  # local night
    segments = [
        Segment(start, day.tz_localize("UTC"), (day + pd.Timedelta(hours=1)).tz_localize("UTC"), 3000.0),
        Segment(start, night.tz_localize("UTC"), (night + pd.Timedelta(hours=1)).tz_localize("UTC"), 1000.0),
    ]
    assert astronomy.get_nightday_ratio(build_segments(segments)) == pytest.approx(1000.0 / 3000.0, rel=1e-6)


@pytest.mark.parametrize("bearing_deg", [0.0, 90.0, 180.0, 45.0], ids=["north", "east", "south", "north-east"])
def test_nightday_ratio_invariant_to_segment_direction(bearing_deg):
    # The ratio uses only the start point (solar offset + sun_time) and dist_meters, never
    # the segment's bearing, so movement in any direction gives the same answer.
    midnight = pd.Timestamp("2024-03-20", tz="UTC")
    segments = [
        Segment(
            EQUATOR,
            midnight + pd.Timedelta(hours=h),
            midnight + pd.Timedelta(hours=h + 1),
            dist,
            bearing_deg=bearing_deg,
        )
        for h, dist in [(2, 1000.0), (12, 3000.0)]  # one night, one day, asymmetric
    ]
    assert astronomy.get_nightday_ratio(build_segments(segments)) == pytest.approx(1000.0 / 3000.0, rel=1e-6)


def test_nightday_ratio_multiday_segment_split():
    # A single gap-bridging segment spanning ~1.6 local days (a tag that went quiet from
    # one evening to the morning two days later) must be apportioned across every day it
    # covers, not dumped onto its start date. Its ratio should equal that of the same
    # segment pre-split by hand into one piece per calendar day, distance shared by time.
    # At lon 0 the local solar day boundary coincides with UTC midnight.
    start_time = pd.Timestamp("2024-03-20 17:00", tz="UTC")
    end_time = pd.Timestamp("2024-03-22 07:00", tz="UTC")
    long_gdf = build_segments([Segment(EQUATOR, start_time, end_time, 3800.0)])

    bounds = [start_time, pd.Timestamp("2024-03-21", tz="UTC"), pd.Timestamp("2024-03-22", tz="UTC"), end_time]
    total_seconds = (end_time - start_time).total_seconds()
    split_gdf = build_segments(
        [
            Segment(EQUATOR, s, e, 3800.0 * (e - s).total_seconds() / total_seconds)
            for s, e in zip(bounds[:-1], bounds[1:])
        ]
    )

    long_ratio = astronomy.get_nightday_ratio(long_gdf)
    assert long_ratio == pytest.approx(astronomy.get_nightday_ratio(split_gdf), rel=1e-9)
    # Splitting keeps the result moderate (the night-heavy evening/morning edge days pull the
    # per-day mean above 1). The unsplit single-boundary calc would instead dump the whole
    # segment onto one day's sunrise/sunset and inflate the ratio into the hundreds.
    assert 1.0 < long_ratio < 10.0


@pytest.mark.parametrize("lat", [-33.0, -10.0, 45.0])
def test_nightday_ratio_symmetric_across_latitudes(lat):
    # Two clear-night + two clear-day 1h segments on an equinox -> ratio 1.0 at any
    # non-polar latitude, in either hemisphere. Real fixtures are all near-equator NH.
    midnight = pd.Timestamp("2024-03-20", tz="UTC")
    start = Point(0.0, lat)
    segments = [
        Segment(start, midnight + pd.Timedelta(hours=h), midnight + pd.Timedelta(hours=h + 1)) for h in [2, 12, 13, 22]
    ]
    assert astronomy.get_nightday_ratio(build_segments(segments)) == pytest.approx(1.0, rel=1e-6)


@pytest.mark.parametrize("date", [datetime(2024, 6, 21), datetime(2024, 12, 21)])
def test_nightday_ratio_polar_returns_nan(date):
    # At lat 80 the sun never sets (June) / never rises (December): sun_time yields NaT,
    # every day drops out, and the ratio must degrade to NaN rather than crash.
    d = pd.Timestamp(date, tz="UTC")
    segments = [Segment(ARCTIC, d + pd.Timedelta(hours=h), d + pd.Timedelta(hours=h + 1)) for h in [2, 6, 12, 18]]
    assert np.isnan(astronomy.get_nightday_ratio(build_segments(segments)))


def test_nightday_ratio_multiday_three_day_split():
    # Same split-equivalence invariant as the ~1.6-day case above, extended to a segment
    # that spans four local dates, to exercise the general per-day explosion.
    start_time = pd.Timestamp("2024-03-19 20:00", tz="UTC")
    end_time = pd.Timestamp("2024-03-22 05:00", tz="UTC")
    long_gdf = build_segments([Segment(EQUATOR, start_time, end_time, 9000.0)])

    midnights = [pd.Timestamp(d, tz="UTC") for d in ["2024-03-20", "2024-03-21", "2024-03-22"]]
    bounds = [start_time, *midnights, end_time]
    total_seconds = (end_time - start_time).total_seconds()
    split_gdf = build_segments(
        [
            Segment(EQUATOR, s, e, 9000.0 * (e - s).total_seconds() / total_seconds)
            for s, e in zip(bounds[:-1], bounds[1:])
        ]
    )
    assert astronomy.get_nightday_ratio(long_gdf) == pytest.approx(astronomy.get_nightday_ratio(split_gdf), rel=1e-9)


def test_nightday_ratio_all_night_returns_inf():
    # A day with movement only at night is fully nocturnal: its night fraction is 1.0, so it
    # is kept (not dropped as it was under the old raw night/day ratio) and, being the only
    # day, maps to an infinite ratio rather than NaN.
    segments = [
        Segment(EQUATOR, pd.Timestamp("2024-03-20 01:00", tz="UTC"), pd.Timestamp("2024-03-20 02:00", tz="UTC")),
        Segment(EQUATOR, pd.Timestamp("2024-03-20 22:00", tz="UTC"), pd.Timestamp("2024-03-20 23:00", tz="UTC")),
    ]
    assert np.isinf(astronomy.get_nightday_ratio(build_segments(segments)))


def test_nightday_ratio_nocturnal_days_not_dropped():
    # Two purely-nocturnal days (no daytime movement) plus one day with a little daytime
    # movement. The nocturnal days must stay in the average as night fraction 1.0 rather than
    # dropping out; the resulting ratio is far more nocturnal than the ~50 the old
    # drop-the-inf-days behaviour reported (it kept only the single mixed day).
    def night(day, dist):
        return Segment(EQUATOR, pd.Timestamp(f"{day} 22:00", tz="UTC"), pd.Timestamp(f"{day} 23:00", tz="UTC"), dist)

    segments = [
        night("2024-03-20", 5000.0),  # pure-night day
        night("2024-03-21", 5000.0),  # pure-night day
        night("2024-03-22", 5000.0),  # mostly-night day ...
        Segment(EQUATOR, pd.Timestamp("2024-03-22 12:00", tz="UTC"), pd.Timestamp("2024-03-22 13:00", tz="UTC"), 100.0),
    ]
    # Fractions: [1.0, 1.0, 5000/5100]; mean 0.99346; ratio m/(1-m).
    mean_fraction = (1.0 + 1.0 + 5000.0 / 5100.0) / 3.0
    expected = mean_fraction / (1.0 - mean_fraction)
    assert astronomy.get_nightday_ratio(build_segments(segments)) == pytest.approx(expected, rel=1e-6)


@pytest.mark.parametrize(
    "segment_start, segment_end, expected, label",
    [
        (datetime(2024, 1, 1, 4), datetime(2024, 1, 1, 20), 0.75, "spans both sunrise and sunset"),
        (datetime(2024, 1, 1, 0), datetime(2024, 1, 2, 0), 0.5, "exactly one full calendar day"),
    ],
)
def test_calculate_day_fraction_full_day(segment_start, segment_end, expected, label):
    # An interval containing both sunrise and sunset must count only the daylight window
    # between them; the previous single-crossing branch logic mis-scored these.
    actual = _day_fraction_one(SUNRISE, SUNSET, segment_start, segment_end)
    assert actual == pytest.approx(expected), f"{label}: got {actual}, expected {expected}"


@pytest.fixture
def daily_summary():
    """Create a sample daily summary DataFrame for testing."""
    date = datetime(2024, 1, 1)
    df = pd.DataFrame(
        {
            "sunrise": [datetime(2024, 1, 1, 6, 0)],
            "sunset": [datetime(2024, 1, 1, 18, 0)],
            "day_distance": [0.0],
            "night_distance": [0.0],
        },
        index=[date],
    )
    return df


def test_all_night_before_sunrise(daily_summary):
    """Test segment entirely before sunrise."""
    date = datetime(2024, 1, 1)
    astronomy.calculate_day_night_distance(
        date, datetime(2024, 1, 1, 2, 0), datetime(2024, 1, 1, 4, 0), 1000, daily_summary
    )  # 2:00 AM  # 4:00 AM
    assert daily_summary.loc[date, "night_distance"] == 1000
    assert daily_summary.loc[date, "day_distance"] == 0


def test_all_night_after_sunset(daily_summary):
    """Test segment entirely after sunset."""
    date = datetime(2024, 1, 1)
    astronomy.calculate_day_night_distance(
        date, datetime(2024, 1, 1, 20, 0), datetime(2024, 1, 1, 22, 0), 1000, daily_summary
    )  # 8:00 PM  # 10:00 PM
    assert daily_summary.loc[date, "night_distance"] == 1000
    assert daily_summary.loc[date, "day_distance"] == 0


def test_all_day(daily_summary):
    """Test segment entirely during daylight hours."""
    date = datetime(2024, 1, 1)
    astronomy.calculate_day_night_distance(
        date, datetime(2024, 1, 1, 10, 0), datetime(2024, 1, 1, 14, 0), 1000, daily_summary
    )  # 10:00 AM  # 2:00 PM
    assert daily_summary.loc[date, "day_distance"] == 1000
    assert daily_summary.loc[date, "night_distance"] == 0


def test_day_to_night_transition(daily_summary):
    """Test segment starting in day and ending in night."""
    date = datetime(2024, 1, 1)
    astronomy.calculate_day_night_distance(
        date, datetime(2024, 1, 1, 17, 0), datetime(2024, 1, 1, 19, 0), 1000, daily_summary
    )  # 5:00 PM  # 7:00 PM
    # Segment spans 2 hours, with 1 hour in day and 1 hour in night
    assert daily_summary.loc[date, "day_distance"] == 500
    assert daily_summary.loc[date, "night_distance"] == 500


def test_night_to_day_transition(daily_summary):
    """Test segment starting in night and ending in day."""
    date = datetime(2024, 1, 1)
    astronomy.calculate_day_night_distance(
        date, datetime(2024, 1, 1, 5, 0), datetime(2024, 1, 1, 7, 0), 1000, daily_summary
    )  # 5:00 AM  # 7:00 AM
    # Segment spans 2 hours, with 1 hour in night and 1 hour in day
    assert daily_summary.loc[date, "day_distance"] == 500
    assert daily_summary.loc[date, "night_distance"] == 500


@pytest.fixture
def inverted_daily_summary():
    """Create a daily summary where sunrise is after sunset in UTC."""
    date = datetime(2024, 1, 1)
    df = pd.DataFrame(
        {
            "sunrise": [datetime(2024, 1, 1, 18, 0)],  # 6:00 PM
            "sunset": [datetime(2024, 1, 1, 6, 0)],  # 6:00 AM
            "day_distance": [0.0],
            "night_distance": [0.0],
        },
        index=[date],
    )
    return df


def test_inverted_all_day(inverted_daily_summary):
    """Test all-day segment when sunrise is after sunset."""
    date = datetime(2024, 1, 1)
    astronomy.calculate_day_night_distance(
        date,
        datetime(2024, 1, 1, 4, 0),
        datetime(2024, 1, 1, 5, 0),
        1000,
        inverted_daily_summary,  # 4:00 AM  # 5:00 AM
    )
    assert inverted_daily_summary.loc[date, "day_distance"] == 1000
    assert inverted_daily_summary.loc[date, "night_distance"] == 0


def test_inverted_all_night(inverted_daily_summary):
    """Test all-night segment when sunrise is after sunset."""
    date = datetime(2024, 1, 1)
    astronomy.calculate_day_night_distance(
        date,
        datetime(2024, 1, 1, 7, 0),
        datetime(2024, 1, 1, 17, 0),
        1000,
        inverted_daily_summary,  # 7:00 AM  # 5:00 PM
    )
    assert inverted_daily_summary.loc[date, "night_distance"] == 1000
    assert inverted_daily_summary.loc[date, "day_distance"] == 0


def _day_fraction_one(sunrise, sunset, segment_start, segment_end):
    result = astronomy.calculate_day_fraction(
        sunrise=pd.Series([sunrise]),
        sunset=pd.Series([sunset]),
        segment_start=pd.Series([segment_start]),
        segment_end=pd.Series([segment_end]),
    )
    return result[0]


@pytest.mark.parametrize(
    "sunrise, sunset, segment_start, segment_end, expected, label",
    [
        # --- normal day (sunrise < sunset) ---
        (SUNRISE, SUNSET, datetime(2024, 1, 1, 17), datetime(2024, 1, 1, 19), 0.5, "normal: day->night transition"),
        (SUNRISE, SUNSET, datetime(2024, 1, 1, 5), datetime(2024, 1, 1, 7), 0.5, "normal: night->day transition"),
        (SUNRISE, SUNSET, datetime(2024, 1, 1, 2), datetime(2024, 1, 1, 4), 0.0, "normal: all night before sunrise"),
        (SUNRISE, SUNSET, datetime(2024, 1, 1, 20), datetime(2024, 1, 1, 22), 0.0, "normal: all night after sunset"),
        (SUNRISE, SUNSET, datetime(2024, 1, 1, 10), datetime(2024, 1, 1, 14), 1.0, "normal: all day"),
        # --- inverted day (sunrise > sunset, polar / high latitude) ---
        (
            INVERTED_SUNRISE,
            INVERTED_SUNSET,
            datetime(2024, 1, 1, 4),
            datetime(2024, 1, 1, 5),
            1.0,
            "inverted: all day before sunset",
        ),
        (
            INVERTED_SUNRISE,
            INVERTED_SUNSET,
            datetime(2024, 1, 1, 20),
            datetime(2024, 1, 1, 22),
            1.0,
            "inverted: all day after sunrise",
        ),
        (
            INVERTED_SUNRISE,
            INVERTED_SUNSET,
            datetime(2024, 1, 1, 7),
            datetime(2024, 1, 1, 17),
            0.0,
            "inverted: all night",
        ),
        (
            INVERTED_SUNRISE,
            INVERTED_SUNSET,
            datetime(2024, 1, 1, 5),
            datetime(2024, 1, 1, 7),
            0.5,
            "inverted: day->night transition at sunset",
        ),
        (
            INVERTED_SUNRISE,
            INVERTED_SUNSET,
            datetime(2024, 1, 1, 17),
            datetime(2024, 1, 1, 19),
            0.5,
            "inverted: night->day transition at sunrise",
        ),
    ],
)
def test_calculate_day_fraction_branches(sunrise, sunset, segment_start, segment_end, expected, label):
    actual = _day_fraction_one(sunrise, sunset, segment_start, segment_end)
    assert actual == pytest.approx(expected), f"{label}: got {actual}, expected {expected}"


@pytest.mark.parametrize(
    "sunrise, sunset, segment_start, segment_end, expected, label",
    [
        # The four boundary edge cases that the strict-inequality version dropped to NaN.
        (SUNRISE, SUNSET, datetime(2024, 1, 1, 18), datetime(2024, 1, 1, 20), 0.0, "normal: start exactly at sunset"),
        (SUNRISE, SUNSET, datetime(2024, 1, 1, 4), datetime(2024, 1, 1, 6), 0.0, "normal: end exactly at sunrise"),
        (
            INVERTED_SUNRISE,
            INVERTED_SUNSET,
            datetime(2024, 1, 1, 18),
            datetime(2024, 1, 1, 20),
            1.0,
            "inverted: start exactly at sunrise",
        ),
        (
            INVERTED_SUNRISE,
            INVERTED_SUNSET,
            datetime(2024, 1, 1, 4),
            datetime(2024, 1, 1, 6),
            1.0,
            "inverted: end exactly at sunset",
        ),
    ],
)
def test_calculate_day_fraction_boundary_edges(sunrise, sunset, segment_start, segment_end, expected, label):
    actual = _day_fraction_one(sunrise, sunset, segment_start, segment_end)
    assert not np.isnan(actual), f"{label}: fell through to NaN"
    assert actual == pytest.approx(expected), f"{label}: got {actual}, expected {expected}"


def test_calculate_day_fraction_vectorized():
    """Run several rows in one call to confirm vectorization preserves alignment."""
    rows = [
        (SUNRISE, SUNSET, datetime(2024, 1, 1, 17), datetime(2024, 1, 1, 19), 0.5),
        (SUNRISE, SUNSET, datetime(2024, 1, 1, 10), datetime(2024, 1, 1, 14), 1.0),
        (SUNRISE, SUNSET, datetime(2024, 1, 1, 2), datetime(2024, 1, 1, 4), 0.0),
        (INVERTED_SUNRISE, INVERTED_SUNSET, datetime(2024, 1, 1, 7), datetime(2024, 1, 1, 17), 0.0),
        (SUNRISE, SUNSET, datetime(2024, 1, 1, 18), datetime(2024, 1, 1, 20), 0.0),
    ]
    sunrise, sunset, starts, ends, expected = zip(*rows)
    actual = astronomy.calculate_day_fraction(
        sunrise=pd.Series(sunrise),
        sunset=pd.Series(sunset),
        segment_start=pd.Series(starts),
        segment_end=pd.Series(ends),
    )
    np.testing.assert_allclose(actual, expected)


def test_sun_time_unresolved_returns_nat():
    # At a polar-day latitude the sun never sets within astroplan's search window, so
    # astroplan returns a masked 0-d value. sun_time must coerce it to NaT rather than
    # passing the 0-d array through, which would later crash get_nightday_ratio with
    # "iteration over a 0-d array" during the calculate_day_fraction comparisons.
    result = astronomy.sun_time(datetime(2025, 6, 21), Point(0.0, 80.0))
    assert pd.isna(result["sunrise"])
    assert pd.isna(result["sunset"])
