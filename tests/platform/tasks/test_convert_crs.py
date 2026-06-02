import geopandas as gpd  # type: ignore[import-untyped]
import pytest
from shapely.geometry import Point, Polygon

from ecoscope.platform.tasks.transformation import convert_crs


def _gdf(crs: str | None = "EPSG:4326") -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        {"id": [1, 2], "geometry": [Point(0, 0), Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])]},
        crs=crs,
    )


def test_convert_crs_reprojects_to_target():
    result = convert_crs(_gdf("EPSG:4326"), crs="EPSG:3857")
    assert result.crs.to_epsg() == 3857


def test_convert_crs_default_is_4326():
    result = convert_crs(_gdf("EPSG:3857"))
    assert result.crs.to_epsg() == 4326


def test_convert_crs_noop_when_already_in_target_crs():
    gdf = _gdf("EPSG:4326")
    result = convert_crs(gdf, crs="EPSG:4326")
    assert result.crs.to_epsg() == 4326
    assert list(result["id"]) == [1, 2]


def test_convert_crs_raises_when_input_has_no_crs():
    with pytest.raises(ValueError, match="no CRS information"):
        convert_crs(_gdf(crs=None), crs="EPSG:4326")
