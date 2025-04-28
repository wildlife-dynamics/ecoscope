from datetime import datetime

import pandas as pd
import pyproj
import pytest

from ecoscope.analysis import astronomy
from ecoscope import Trajectory


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


def test_nightday_ratio(movebank_relocations):
    trajectory = Trajectory.from_relocations(movebank_relocations)
    expected = pd.Series(
        [0.45905845612291696, 2.0019632541788472],
        index=pd.Index(["Habiba", "Salif Keita"], name="groupby_col"),
    )
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
