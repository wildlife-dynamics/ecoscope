import os

import geopandas as gpd
from shapely.geometry import LineString, Point

import ecoscope


def test_feature_density_point():

    AOI = gpd.read_file(os.path.join("tests/sample_data/vector", "landscape_grid.gpkg"), layer="AOI")

    grid = gpd.GeoDataFrame(
        geometry=ecoscope.base.utils.create_meshgrid(
            AOI.unary_union, in_crs=AOI.crs, out_crs=4326, xlen=5000, ylen=5000, return_intersecting_only=False
        )
    )

    points = gpd.GeoDataFrame(geometry=[Point(34.72, -2.1), Point(34.73, -2.1), Point(36.06, -1.13)], crs=4326)

    density = ecoscope.analysis.calculate_feature_density(points, grid, geometry_type="point")

    print(density)


def test_feature_density_line():

    AOI = gpd.read_file(os.path.join("tests/sample_data/vector", "landscape_grid.gpkg"), layer="AOI")

    grid = gpd.GeoDataFrame(
        geometry=ecoscope.base.utils.create_meshgrid(
            AOI.unary_union, in_crs=AOI.crs, out_crs=4326, xlen=5000, ylen=5000, return_intersecting_only=False
        )
    )

    lines = gpd.GeoDataFrame(
        geometry=[LineString([(34.72, -2.1), (34.73, -2.1)]), LineString([(36.06, -1.13), (34.72, -2.1)])], crs=4326
    )

    density = ecoscope.analysis.calculate_feature_density(lines, grid, geometry_type="line")

    print(density)
