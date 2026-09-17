import os
from datetime import datetime, timezone
from importlib.resources import files

import geopandas as gpd  # type: ignore[import-untyped]
import pytest

from ecoscope.platform.connections import EarthEngineConnection
from ecoscope.platform.tasks.analysis import (
    create_forest_layers,
    extract_forest_cover_trends,
)
from ecoscope.platform.tasks.analysis._hansen_forest_change import _parse_year_range
from ecoscope.platform.tasks.filter._filter import UTC_TIMEZONEINFO, TimeRange
from ecoscope.platform.tasks.results._pydeck import BitmapLayerDefinition

_HANSEN_IMAGE = "UMD/hansen/global_forest_change_2024_v1_12"


def test_parse_year_range_none():
    assert _parse_year_range(None) == (None, None)


def test_parse_year_range_extracts_calendar_years():
    time_range = TimeRange(
        since=datetime(2010, 6, 1, tzinfo=timezone.utc),
        until=datetime(2020, 3, 1, tzinfo=timezone.utc),
        timezone=UTC_TIMEZONEINFO,
    )
    assert _parse_year_range(time_range) == (2010, 2020)


@pytest.fixture
def client():
    return EarthEngineConnection(
        service_account=os.environ["EE_SERVICE_ACCOUNT"],
        private_key_file=os.environ["EE_PRIVATE_KEY_FILE"],
    ).get_client()


@pytest.fixture
def roi():
    example_input_df_path = files("ecoscope.platform.tasks.io") / "download-roi.example-return.parquet"
    return gpd.read_parquet(example_input_df_path).loc[["Mara / Serengeti"]].reset_index(drop=True)


@pytest.mark.io
def test_extract_forest_cover_trends(client, roi):
    result = extract_forest_cover_trends(
        client=client,
        aoi=roi,
        image=_HANSEN_IMAGE,
        time_range=TimeRange(
            since=datetime(2015, 1, 1, tzinfo=timezone.utc),
            until=datetime(2020, 12, 31, tzinfo=timezone.utc),
            timezone=UTC_TIMEZONEINFO,
        ),
    )

    assert list(result.columns) == ["year", "loss_area", "cumsum_loss_area", "survival_area", "cumsum_loss_pct"]
    assert result["year"].between(2015, 2020).all()
    assert (result["cumsum_loss_area"].diff().dropna() >= 0).all()


@pytest.mark.io
def test_create_forest_layers(client, roi):
    result = create_forest_layers(
        client=client,
        aoi=roi,
        image=_HANSEN_IMAGE,
        time_range=TimeRange(
            since=datetime(2015, 1, 1, tzinfo=timezone.utc),
            until=datetime(2020, 12, 31, tzinfo=timezone.utc),
            timezone=UTC_TIMEZONEINFO,
        ),
        opacity=0.8,
    )

    assert isinstance(result, BitmapLayerDefinition)
    assert result.image.startswith("data:image/png;base64,")
    assert result.opacity == 0.8
    assert len(result.bounds) == 4
    assert result.legend is not None
    assert [v.label for v in result.legend.values] == ["Forest Cover", "Forest Loss"]
