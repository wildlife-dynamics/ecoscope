import geopandas as gpd  # type: ignore[import-untyped]

from ecoscope.platform.tasks.analysis import (
    calculate_feature_density,
    create_meshgrid,
)
from ecoscope.platform.tasks.analysis._time_density import (
    CustomGridCellSize,
)

from ..utils.random_geometry import random_3857_rectangle, random_points_in_bounds


def test_calculate_feature_density():
    bounds = random_3857_rectangle(500, 500, 500, 500, utm_safe=True)

    aoi = gpd.GeoDataFrame(geometry=[bounds], crs="EPSG:3857")
    meshgrid = create_meshgrid(
        aoi,
        auto_scale_or_custom_cell_size=CustomGridCellSize(grid_cell_size=100),
    )
    random_points = random_points_in_bounds(bounds, 200)
    points_gdf = gpd.GeoDataFrame(geometry=random_points, crs="EPSG:3857")

    result = calculate_feature_density(
        geodataframe=points_gdf,
        meshgrid=meshgrid,
        geometry_type="point",
    )

    # the sum of all density vals should equal our number of points
    assert result.density.sum() == 200


def test_calculate_feature_density_empty_sum_column_counts():
    bounds = random_3857_rectangle(500, 500, 500, 500, utm_safe=True)

    aoi = gpd.GeoDataFrame(geometry=[bounds], crs="EPSG:3857")
    meshgrid = create_meshgrid(
        aoi,
        auto_scale_or_custom_cell_size=CustomGridCellSize(grid_cell_size=100),
    )
    random_points = random_points_in_bounds(bounds, 200)
    points_gdf = gpd.GeoDataFrame(geometry=random_points, crs="EPSG:3857")

    result = calculate_feature_density(
        geodataframe=points_gdf,
        meshgrid=meshgrid,
        geometry_type="point",
        sum_column="",
    )

    # empty string behaves like None: count rows per cell
    assert result.density.sum() == 200


def test_calculate_feature_density_sum_column_coerced_to_numeric():
    bounds = random_3857_rectangle(500, 500, 500, 500, utm_safe=True)

    aoi = gpd.GeoDataFrame(geometry=[bounds], crs="EPSG:3857")
    meshgrid = create_meshgrid(
        aoi,
        auto_scale_or_custom_cell_size=CustomGridCellSize(grid_cell_size=100),
    )
    random_points = random_points_in_bounds(bounds, 200)
    points_gdf = gpd.GeoDataFrame(geometry=random_points, crs="EPSG:3857")
    # string-typed numbers, with one non-castable value that should be ignored
    points_gdf["Number of Animals"] = ["2"] * 199 + ["n/a"]

    result = calculate_feature_density(
        geodataframe=points_gdf,
        meshgrid=meshgrid,
        geometry_type="point",
        sum_column="Number of Animals",
    )

    assert result.density.sum() == 398


def test_calculate_feature_density_missing_sum_column_raises():
    import pytest

    bounds = random_3857_rectangle(500, 500, 500, 500, utm_safe=True)

    aoi = gpd.GeoDataFrame(geometry=[bounds], crs="EPSG:3857")
    meshgrid = create_meshgrid(
        aoi,
        auto_scale_or_custom_cell_size=CustomGridCellSize(grid_cell_size=100),
    )
    random_points = random_points_in_bounds(bounds, 10)
    points_gdf = gpd.GeoDataFrame(geometry=random_points, crs="EPSG:3857")

    with pytest.raises(ValueError, match="not found in the feature data"):
        calculate_feature_density(
            geodataframe=points_gdf,
            meshgrid=meshgrid,
            geometry_type="point",
            sum_column="No Such Column",
        )
