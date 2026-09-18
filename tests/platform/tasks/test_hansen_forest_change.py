import os
from datetime import datetime, timezone
from importlib.resources import files

import geopandas as gpd  # type: ignore[import-untyped]
import pytest
from shapely.geometry import Polygon

from ecoscope.platform.connections import EarthEngineConnection
from ecoscope.platform.tasks.analysis import (
    create_forest_layers,
    extract_forest_cover_trends,
)
from ecoscope.platform.tasks.analysis._hansen_forest_change import _ensure_wgs84, _parse_year_range
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


def _square(x: float, y: float) -> Polygon:
    return Polygon([(x, y), (x + 1, y), (x + 1, y + 1), (x, y + 1)])


def test_ensure_wgs84_assumes_wgs84_when_crs_missing():
    gdf = gpd.GeoDataFrame({"geometry": [_square(0, 0)]})
    assert gdf.crs is None

    result = _ensure_wgs84(gdf)

    assert result.crs.to_epsg() == 4326


def test_ensure_wgs84_reprojects_non_wgs84_crs():
    gdf = gpd.GeoDataFrame({"geometry": [_square(0, 0)]}, crs="EPSG:3857")

    result = _ensure_wgs84(gdf)

    assert result.crs.to_epsg() == 4326


def test_ensure_wgs84_passthrough_when_already_wgs84():
    gdf = gpd.GeoDataFrame({"geometry": [_square(0, 0)]}, crs="EPSG:4326")

    result = _ensure_wgs84(gdf)

    assert result is gdf


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
def test_extract_forest_cover_trends_propagates_name_column(client, roi):
    """If `aoi` carries a "name" column (e.g. from load_spatial_features_group),
    it should be propagated onto every output row - GAMM's combined-fit site
    pooling (see _trend_analysis.py) depends on this to tell regions apart
    after multiple regions' results are concatenated back together."""
    named_roi = roi.assign(name="Mara / Serengeti")

    result = extract_forest_cover_trends(
        client=client,
        aoi=named_roi,
        image=_HANSEN_IMAGE,
        time_range=TimeRange(
            since=datetime(2015, 1, 1, tzinfo=timezone.utc),
            until=datetime(2020, 12, 31, tzinfo=timezone.utc),
            timezone=UTC_TIMEZONEINFO,
        ),
    )

    assert list(result.columns) == [
        "year",
        "loss_area",
        "cumsum_loss_area",
        "survival_area",
        "cumsum_loss_pct",
        "name",
    ]
    assert (result["name"] == "Mara / Serengeti").all()


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
