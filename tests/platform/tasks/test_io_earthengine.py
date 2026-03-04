import os
from datetime import datetime, timezone
from importlib.resources import files

import geopandas as gpd  # type: ignore[import-untyped]
import pytest
from ecoscope.platform.tasks.filter._filter import UTC_TIMEZONEINFO, TimeRange
from ecoscope.platform.connections import EarthEngineConnection
from ecoscope.platform.tasks.io import (
    calculate_ndvi_range,
    determine_season_windows,
)

pytestmark = pytest.mark.io


@pytest.fixture
def client():
    return EarthEngineConnection(
        service_account=os.environ["EE_SERVICE_ACCOUNT"],
        private_key_file=os.environ["EE_PRIVATE_KEY_FILE"],
    ).get_client()


def test_calculate_ndvi_range(client):
    example_input_df_path = (
        files("ecoscope.platform.tasks.io")
        / "download-roi.example-return.parquet"
    )
    roi = (
        gpd.read_parquet(example_input_df_path)
        .loc[["Mara / Serengeti"]]
        .reset_index(drop=True)
    )
    result = calculate_ndvi_range(
        client=client,
        time_range=TimeRange(
            since=datetime.strptime("2023-01-01", "%Y-%m-%d").replace(
                tzinfo=timezone.utc
            ),
            until=datetime.strptime("2023-12-31", "%Y-%m-%d").replace(
                tzinfo=timezone.utc
            ),
            timezone=UTC_TIMEZONEINFO,
        ),
        roi=roi,
        img_coll_name="MODIS/061/MYD13A1",
        band="NDVI",
        scale_factor=0.0001,
        analysis_scale=500.0,
    )

    assert len(result) > 0
    assert "min" in result
    assert "max" in result
    assert "mean" in result
    assert "NDVI" in result
    assert "img_date" in result


def test_determine_season_windows(client):
    example_input_df_path = (
        files("ecoscope.platform.tasks.io")
        / "download-roi.example-return.parquet"
    )
    roi = (
        gpd.read_parquet(example_input_df_path)
        .loc[["Mara / Serengeti"]]
        .reset_index(drop=True)
    )
    result = determine_season_windows(
        client=client,
        time_range=TimeRange(
            since=datetime.strptime("2023-01-01", "%Y-%m-%d").replace(
                tzinfo=timezone.utc
            ),
            until=datetime.strptime("2023-01-31", "%Y-%m-%d").replace(
                tzinfo=timezone.utc
            ),
            timezone=UTC_TIMEZONEINFO,
        ),
        roi=roi,
    )

    assert len(result) > 0
    assert "start" in result
    assert "end" in result
    assert "season" in result
