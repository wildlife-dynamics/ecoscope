import math
import os

import geopandas as gpd
import pytest
from shapely.geometry import LineString, Point

import ecoscope
from ecoscope.analysis.feature_density import calculate_feature_density


def test_feature_density_point():
    AOI = gpd.read_file(os.path.join("tests/sample_data/vector", "landscape_grid.gpkg"), layer="AOI")

    grid = gpd.GeoDataFrame(
        geometry=ecoscope.base.utils.create_meshgrid(
            AOI.union_all(), in_crs=AOI.crs, out_crs="EPSG:3857", xlen=5000, ylen=5000, return_intersecting_only=False
        )
    )

    points = gpd.GeoDataFrame(
        geometry=[
            Point(3906043.099, -143047.752),
            Point(3957465.506, -178334.298),
            Point(3957936.393, -178331.41),
        ],
        crs="EPSG:3857",
    )

    density_grid = calculate_feature_density(points, grid, geometry_type="point")
    assert density_grid["density"].sum() == 3
    assert density_grid["density"].max() == 2

    ecoscope.io.raster.grid_to_raster(
        density_grid,
        xlen=5000,
        ylen=5000,
        val_column="density",
        out_dir="tests/test_output",
        raster_name="point_density.tif",
    )


@pytest.mark.parametrize("aggregate_function", ["sum", "mean"])
def test_feature_density_point_values(aggregate_function):
    AOI = gpd.read_file(os.path.join("tests/sample_data/vector", "landscape_grid.gpkg"), layer="AOI")

    grid = gpd.GeoDataFrame(
        geometry=ecoscope.base.utils.create_meshgrid(
            AOI.union_all(), in_crs=AOI.crs, out_crs="EPSG:3857", xlen=5000, ylen=5000, return_intersecting_only=False
        )
    )

    points = gpd.GeoDataFrame(
        data={"values": [2, 500, 34]},
        geometry=[
            Point(3906043.099, -143047.752),
            Point(3957465.506, -178334.298),
            Point(3957936.393, -178331.41),
        ],
        crs="EPSG:3857",
    )

    density_grid = calculate_feature_density(
        points, grid, geometry_type="point", aggregate_function=aggregate_function, aggregate_column="values"
    )

    # In these tests the final two points fall in the same grid cell
    if aggregate_function == "sum":
        assert density_grid["density"].sum() == 536
        assert density_grid["density"].max() == 534
    if aggregate_function == "mean":
        assert density_grid["density"].sum() == 269
        assert density_grid["density"].max() == 267


def test_feature_density_line():
    AOI = gpd.read_file(os.path.join("tests/sample_data/vector", "landscape_grid.gpkg"), layer="AOI")

    grid = gpd.GeoDataFrame(
        geometry=ecoscope.base.utils.create_meshgrid(
            AOI.union_all(), in_crs=AOI.crs, out_crs="EPSG:3857", xlen=5000, ylen=5000, return_intersecting_only=False
        )
    )

    lines = gpd.GeoDataFrame(
        geometry=[
            LineString([(3906043.099, -143047.752), (3957465.506, -178334.298)]),
            LineString([(3957936.393, -178331.41), (3957465.506, -178334.298)]),
        ],
        crs="EPSG:3857",
    )

    density_grid = calculate_feature_density(lines, grid, geometry_type="line")

    assert math.isclose(density_grid["density"].sum(), 62835.98440960531)
    assert math.isclose(density_grid["density"].max(), 6063.999352799931)
    ecoscope.io.raster.grid_to_raster(
        density_grid,
        xlen=5000,
        ylen=5000,
        val_column="density",
        out_dir="tests/test_output",
        raster_name="line_density.tif",
    )


@pytest.mark.parametrize(
    "aggregate_function, aggregate_column",
    [
        ("sum", "values"),
        ("mean", "values"),
    ],
)
def test_feature_density_line_with_values(aggregate_function, aggregate_column):
    AOI = gpd.read_file(os.path.join("tests/sample_data/vector", "landscape_grid.gpkg"), layer="AOI")

    grid = gpd.GeoDataFrame(
        geometry=ecoscope.base.utils.create_meshgrid(
            AOI.union_all(), in_crs=AOI.crs, out_crs="EPSG:3857", xlen=5000, ylen=5000, return_intersecting_only=False
        )
    )

    lines = gpd.GeoDataFrame(
        data={"values": [300, 200]},
        geometry=[
            LineString([(3906043.099, -143047.752), (3957465.506, -178334.298)]),
            LineString([(3957936.393, -178331.41), (3957465.506, -178334.298)]),
        ],
        crs="EPSG:3857",
    )

    density_grid = calculate_feature_density(
        lines, grid, geometry_type="line", aggregate_function=aggregate_function, aggregate_column=aggregate_column
    )

    if aggregate_function == "sum":
        assert density_grid["density"].sum() == 536
        assert density_grid["density"].max() == 534
    if aggregate_function == "mean":
        assert density_grid["density"].sum() == 269
        assert density_grid["density"].max() == 267
