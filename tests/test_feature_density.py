import math
import os

import geopandas as gpd
from shapely.geometry import LineString, Point

import ecoscope
from ecoscope.analysis.feature_density import calculate_feature_density


def test_feature_density_point():

    AOI = gpd.read_file(os.path.join("tests/sample_data/vector", "landscape_grid.gpkg"), layer="AOI")
    my_crs = 32736

    grid = gpd.GeoDataFrame(
        geometry=ecoscope.base.utils.create_meshgrid(
            AOI.unary_union, in_crs=AOI.crs, out_crs=my_crs, xlen=5000, ylen=5000, return_intersecting_only=False
        )
    )

    points = gpd.GeoDataFrame(
        geometry=[Point(694000, 9765400), Point(694500, 9765400), Point(842000, 9874000)], crs=my_crs
    )

    density_grid = calculate_feature_density(points, grid, geometry_type="point")
    assert density_grid["density"].sum() == 3
    assert density_grid["density"].max() == 2

    ecoscope.io.raster.grid_to_raster(
        density_grid, val_column="density", out_dir="tests/test_output", raster_name="point_density.tif"
    )


def test_feature_density_line():

    AOI = gpd.read_file(os.path.join("tests/sample_data/vector", "landscape_grid.gpkg"), layer="AOI")
    my_crs = 32736

    grid = gpd.GeoDataFrame(
        geometry=ecoscope.base.utils.create_meshgrid(
            AOI.unary_union, in_crs=AOI.crs, out_crs=my_crs, xlen=5000, ylen=5000, return_intersecting_only=False
        )
    )

    lines = gpd.GeoDataFrame(
        geometry=[
            LineString([(694000, 9765400), (694500, 9765400)]),
            LineString([(842000, 9874000), (694500, 9765400)]),
        ],
        crs=my_crs,
    )

    density_grid = calculate_feature_density(lines, grid, geometry_type="line")

    assert math.isclose(density_grid["density"].sum(), 183667.16408789358)
    assert math.isclose(density_grid["density"].max(), 6209.056409759374)
    ecoscope.io.raster.grid_to_raster(
        density_grid, val_column="density", out_dir="tests/test_output", raster_name="line_density.tif"
    )
