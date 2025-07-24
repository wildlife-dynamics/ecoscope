import math
import os

import geopandas as gpd
from shapely.geometry import LineString, Point

import ecoscope
from ecoscope.analysis.feature_density import calculate_feature_density


def test_feature_density_point():
    AOI = gpd.read_file(os.path.join("tests/sample_data/vector", "landscape_grid.gpkg"), layer="AOI")
    my_crs = "EPSG:4326"

    grid = gpd.GeoDataFrame(
        geometry=ecoscope.base.utils.create_meshgrid(
            AOI.unary_union, in_crs=AOI.crs, out_crs=my_crs, xlen=5000, ylen=5000, return_intersecting_only=False
        )
    )

    points = gpd.GeoDataFrame(
        geometry=[
            Point(35.088582161126965, -1.2849121042736442),
            Point(35.550517503980956, -1.6017955576824079),
            Point(35.55474755114645, -1.601769625072393),
        ],
        crs=my_crs,
    )

    density_grid = calculate_feature_density(points, grid, geometry_type="point")
    assert density_grid["density"].sum() == 3
    assert density_grid["density"].max() == 2

    ecoscope.io.raster.grid_to_raster(
        density_grid, val_column="density", out_dir="tests/test_output", raster_name="point_density.tif"
    )


def test_feature_density_line():
    AOI = gpd.read_file(os.path.join("tests/sample_data/vector", "landscape_grid.gpkg"), layer="AOI")
    my_crs = "EPSG:4326"

    grid = gpd.GeoDataFrame(
        geometry=ecoscope.base.utils.create_meshgrid(
            AOI.unary_union, in_crs=AOI.crs, out_crs=my_crs, xlen=5000, ylen=5000, return_intersecting_only=False
        )
    )

    lines = gpd.GeoDataFrame(
        geometry=[
            LineString([(35.088582161126965, -1.2849121042736442), (35.550517503980956, -1.6017955576824079)]),
            LineString([(35.55474755114645, -1.601769625072393), (35.550517503980956, -1.6017955576824079)]),
        ],
        crs=my_crs,
    )

    density_grid = calculate_feature_density(lines, grid, geometry_type="line")

    assert math.isclose(density_grid["density"].sum(), 0.5644081198166275)
    assert math.isclose(density_grid["density"].max(), 0.05446827796016657)
    ecoscope.io.raster.grid_to_raster(
        density_grid, val_column="density", out_dir="tests/test_output", raster_name="line_density.tif"
    )
