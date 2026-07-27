import geopandas as gpd
import numpy as np
import pytest
from pydantic import TypeAdapter
from shapely.geometry import Point

from ecoscope.platform.tasks.analysis import McpReturnGDF, calculate_minimum_convex_polygon


@pytest.fixture
def relocations_gdf():
    """A tight 21x21 grid of fixes (1m spacing) plus two fixes ~10km away, so percentile
    cutoffs below 100% are guaranteed to exclude exactly the two distant outliers."""
    grid_x, grid_y = np.meshgrid(np.arange(-10, 11), np.arange(-10, 11))
    core_points = [Point(x, y) for x, y in zip(grid_x.ravel(), grid_y.ravel())]
    outliers = [Point(10_000, 10_000), Point(-10_000, -10_000)]
    return gpd.GeoDataFrame(geometry=core_points + outliers, crs="ESRI:102022")


def test_calculate_minimum_convex_polygon_default_percentiles(relocations_gdf):
    result = calculate_minimum_convex_polygon(relocations_gdf, crs="ESRI:102022")

    assert list(result["percentile"]) == [99.999, 95.0, 90.0, 80.0, 70.0, 60.0, 50.0]
    expected_columns = ["percentile", "actual_percentile", "geometry", "area_sqkm"]
    assert all(column in result for column in expected_columns)
    # subject_id is dropped at the task layer, matching calculate_elliptical_time_density's own convention
    assert "subject_id" not in result
    ta = TypeAdapter(McpReturnGDF)
    ta.validate_python(result)


def test_calculate_minimum_convex_polygon_custom_percentiles(relocations_gdf):
    result = calculate_minimum_convex_polygon(relocations_gdf, crs="ESRI:102022", percentiles=[50.0, 90.0, 100.0])

    assert list(result["percentile"]) == [100.0, 90.0, 50.0]
    areas = result.set_index("percentile")["area_sqkm"]
    assert areas.loc[50.0] < areas.loc[90.0] < areas.loc[100.0]
    ta = TypeAdapter(McpReturnGDF)
    ta.validate_python(result)


def test_calculate_minimum_convex_polygon_excludes_outliers_below_100_percent(relocations_gdf):
    result = calculate_minimum_convex_polygon(relocations_gdf, crs="ESRI:102022", percentiles=[99.0])
    hull = result.iloc[0].geometry
    assert not hull.contains(Point(10_000, 10_000))
    assert not hull.contains(Point(-10_000, -10_000))


def test_calculate_minimum_convex_polygon_raises_on_empty_percentiles(relocations_gdf):
    with pytest.raises(ValueError):
        calculate_minimum_convex_polygon(relocations_gdf, crs="ESRI:102022", percentiles=[])


@pytest.mark.parametrize(
    "percentiles",
    [
        [50.0, 90.0, 100.0],
        [100.0, 90.0, 90.0, 50.0],  # duplicates should be deduplicated
    ],
)
def test_calculate_minimum_convex_polygon_deduplicates_percentiles(relocations_gdf, percentiles):
    result = calculate_minimum_convex_polygon(relocations_gdf, crs="ESRI:102022", percentiles=percentiles)
    assert list(result["percentile"]) == [100.0, 90.0, 50.0]
