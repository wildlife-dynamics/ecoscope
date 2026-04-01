import os
from datetime import datetime, timezone
from importlib.resources import files

import geopandas as gpd  # type: ignore[import-untyped]
import pandas as pd
import pytest

from ecoscope.platform.connections import EarthEngineConnection
from ecoscope.platform.tasks.filter._filter import UTC_TIMEZONEINFO, BaselineTimeRange, TimeRange
from ecoscope.platform.tasks.io import (
    calculate_ndvi_range,
    determine_season_windows,
)
from ecoscope.platform.tasks.io._earthengine import _validate_modis_start_date

pytestmark = pytest.mark.io


@pytest.mark.parametrize(
    "doy, expected_period",
    [
        (1, 0),  # first day of year -> period 0
        (16, 0),  # last day of period 0
        (17, 1),  # first day of period 1
        (32, 1),  # last day of period 1
        (33, 2),  # first day of period 2
        (353, 22),  # first day of last period
        (365, 22),  # last day of non-leap year
        (366, 22),  # last day of leap year
    ],
)
def test_modis_16_day_period_number(doy, expected_period):
    """MODIS 16-day period formula maps DOY to correct period (0-22)."""
    assert (doy - 1) // 16 == expected_period


def test_modis_16_day_same_period_across_years():
    """Days in the same MODIS period across different years get the same key."""
    dates = pd.to_datetime(["2020-01-20", "2021-01-20", "2022-01-20"])
    keys = (dates.dayofyear - 1) // 16
    assert keys.nunique() == 1
    assert keys[0] == 1


def test_modis_16_day_adjacent_periods_differ():
    """DOY 16 (period 0) and DOY 17 (period 1) map to different periods."""
    dates = pd.to_datetime(["2023-01-16", "2023-01-17"])
    keys = (dates.dayofyear - 1) // 16
    assert keys[0] == 0
    assert keys[1] == 1


def test_modis_16_day_total_periods_in_year():
    """A full year has 23 MODIS 16-day periods (0-22)."""
    all_days = pd.date_range("2023-01-01", "2023-12-31")
    periods = (all_days.dayofyear - 1) // 16
    assert periods.min() == 0
    assert periods.max() == 22
    assert periods.nunique() == 23


def test_validate_modis_start_date_before_feb_2000():
    """Historical time range cannot start before Feb 2000 (MODIS data availability)."""
    with pytest.raises(ValueError, match="cannot start before 2000-02-01"):
        _validate_modis_start_date(
            TimeRange(
                since=datetime(1999, 1, 1, tzinfo=timezone.utc),
                until=datetime(2020, 12, 31, tzinfo=timezone.utc),
                timezone=UTC_TIMEZONEINFO,
            )
        )


def test_validate_modis_start_date_valid():
    """Valid historical time range starting after Feb 2000."""
    time_range = TimeRange(
        since=datetime(2000, 2, 1, tzinfo=timezone.utc),
        until=datetime(2020, 12, 31, tzinfo=timezone.utc),
        timezone=UTC_TIMEZONEINFO,
    )
    result = _validate_modis_start_date(time_range)
    assert result == time_range


@pytest.fixture
def client():
    return EarthEngineConnection(
        service_account=os.environ["EE_SERVICE_ACCOUNT"],
        private_key_file=os.environ["EE_PRIVATE_KEY_FILE"],
    ).get_client()


@pytest.mark.parametrize("grouping_unit", ["month", "week", "day_of_year", "modis_16_day"])
@pytest.mark.parametrize(
    "ndvi_method",
    ["MODIS MYD13A1 16-Day Composite", "MODIS MCD43A4 Daily NBAR"],
)
def test_calculate_ndvi_range_mode_a(client, grouping_unit, ndvi_method):
    """Mode A: time_range + image_size (default)."""
    example_input_df_path = files("ecoscope_workflows_ext_ecoscope.tasks.io") / "download-roi.example-return.parquet"
    roi = gpd.read_parquet(example_input_df_path).loc[["Mara / Serengeti"]].reset_index(drop=True)
    result = calculate_ndvi_range(
        client=client,
        time_range=TimeRange(
            since=datetime(2023, 1, 1, tzinfo=timezone.utc),
            until=datetime(2023, 12, 31, tzinfo=timezone.utc),
            timezone=UTC_TIMEZONEINFO,
        ),
        roi=roi,
        ndvi_method=ndvi_method,
        grouping_unit=grouping_unit,
    )

    assert len(result) > 0
    assert "min" in result
    assert "max" in result
    assert "mean" in result
    assert "NDVI" in result
    assert "img_date" in result


@pytest.mark.parametrize("grouping_unit", ["month", "week", "day_of_year", "modis_16_day"])
@pytest.mark.parametrize(
    "ndvi_method",
    ["MODIS MYD13A1 16-Day Composite", "MODIS MCD43A4 Daily NBAR"],
)
def test_calculate_ndvi_range_mode_b(client, grouping_unit, ndvi_method):
    """Mode B: baseline_time_range with three dates."""
    example_input_df_path = files("ecoscope_workflows_ext_ecoscope.tasks.io") / "download-roi.example-return.parquet"
    roi = gpd.read_parquet(example_input_df_path).loc[["Mara / Serengeti"]].reset_index(drop=True)
    result = calculate_ndvi_range(
        client=client,
        time_range=TimeRange(
            since=datetime(2023, 1, 1, tzinfo=timezone.utc),
            until=datetime(2023, 12, 31, tzinfo=timezone.utc),
            timezone=UTC_TIMEZONEINFO,
        ),
        baseline_time_range=BaselineTimeRange(
            historical_start=datetime(2015, 1, 1, tzinfo=timezone.utc),
            current_start=datetime(2023, 1, 1, tzinfo=timezone.utc),
            current_end=datetime(2023, 12, 31, tzinfo=timezone.utc),
        ),
        roi=roi,
        ndvi_method=ndvi_method,
        grouping_unit=grouping_unit,
    )

    assert len(result) > 0
    assert "min" in result
    assert "max" in result
    assert "mean" in result
    assert "NDVI" in result
    assert "img_date" in result


def test_determine_season_windows(client):
    example_input_df_path = files("ecoscope_workflows_ext_ecoscope.tasks.io") / "download-roi.example-return.parquet"
    roi = gpd.read_parquet(example_input_df_path).loc[["Mara / Serengeti"]].reset_index(drop=True)
    result = determine_season_windows(
        client=client,
        time_range=TimeRange(
            since=datetime.strptime("2023-01-01", "%Y-%m-%d").replace(tzinfo=timezone.utc),
            until=datetime.strptime("2023-01-31", "%Y-%m-%d").replace(tzinfo=timezone.utc),
            timezone=UTC_TIMEZONEINFO,
        ),
        roi=roi,
    )

    assert len(result) > 0
    assert "start" in result
    assert "end" in result
    assert "season" in result
