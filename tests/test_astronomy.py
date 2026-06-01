from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pyproj
import pytest
from shapely.geometry import Point

from ecoscope import Trajectory
from ecoscope.analysis import astronomy

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
        timezone(timedelta(hours=5)),
        timezone.utc,
        timezone(timedelta(hours=-5)),
    ],
)
def test_nightday_ratio(movebank_relocations, timezone):
    # movebank_relocations is subsampled to keep execution speed low.
    # Expected ratios for the full data are:
    # Habiba=0.45905845612291696, Salif Keita=2.0019632541788472.
    movebank_relocations.gdf = movebank_relocations.gdf.groupby("groupby_col", group_keys=False).head(100)

    trajectory = Trajectory.from_relocations(movebank_relocations)
    expected = pd.Series(
        [0.3736601604553539, 2.1840195829850435],
        index=pd.Index(["Habiba", "Salif Keita"], name="groupby_col"),
    )
    trajectory.gdf["segment_start"] = trajectory.gdf["segment_start"].dt.tz_convert(timezone).dt.as_unit("ns")
    pd.testing.assert_series_equal(
        trajectory.gdf.groupby("groupby_col")[trajectory.gdf.columns].apply(
            astronomy.get_nightday_ratio, include_groups=False
        ),
        expected,
    )


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
