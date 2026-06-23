import geopandas as gpd  # type: ignore[import-untyped]
import pytest
from shapely.geometry import LineString

from ecoscope.platform.tasks.analysis import calculate_patrol_coverage
from ecoscope.platform.tasks.analysis._patrol_coverage import _coverage_area_km2


@pytest.fixture
def trajectory_gdf():
    # Two rangers, each with two overlapping LineString segments near the equator.
    return gpd.GeoDataFrame(
        {
            "ranger": ["A", "A", "B", "B"],
            "geometry": [
                LineString([(0.0, 0.0), (0.1, 0.0)]),
                LineString([(0.05, 0.0), (0.15, 0.0)]),  # overlaps the first
                LineString([(1.0, 1.0), (1.1, 1.0)]),
                LineString([(1.1, 1.0), (1.2, 1.0)]),
            ],
        },
        crs="EPSG:4326",
    )


def test_calculate_patrol_coverage_merged_le_unmerged(trajectory_gdf):
    merged = calculate_patrol_coverage(trajectory_gdf, mode="merged")
    unmerged = calculate_patrol_coverage(trajectory_gdf, mode="unmerged")

    assert merged > 0
    assert merged <= unmerged


def test_calculate_patrol_coverage_scales_with_swath(trajectory_gdf):
    base = calculate_patrol_coverage(trajectory_gdf, swath_width_meters=500.0, mode="unmerged")
    doubled = calculate_patrol_coverage(trajectory_gdf, swath_width_meters=1000.0, mode="unmerged")

    # Buffered-line area is dominated by length * width, so doubling the swath
    # roughly doubles the covered area.
    assert doubled == pytest.approx(2 * base, rel=0.05)


def test_calculate_patrol_coverage_empty():
    empty = gpd.GeoDataFrame({"geometry": []}, crs="EPSG:4326")
    assert _coverage_area_km2(empty, 500.0, merged=True, area_crs="EPSG:6933") == 0.0
