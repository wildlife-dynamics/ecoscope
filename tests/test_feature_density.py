import os

import geopandas as gpd
from shapely.geometry import LineString, Point

import ecoscope

# @pytest.fixture
# def sample_data():
#     # Create sample point data
#     points = gpd.GeoDataFrame(geometry=[
#         Point(1, 1), Point(2, 2), Point(3, 3), Point(4, 4)
#     ], crs="EPSG:4326")

#     # Create sample line data
#     lines = gpd.GeoDataFrame(geometry=[
#         LineString([(1, 1), (2, 2)]), LineString([(3, 3), (4, 4)])
#     ], crs="EPSG:4326")

#     AOI = gpd.read_file(os.path.join("tests/sample_data/vector", "landscape_grid.gpkg"), layer="AOI")

#     grid = gpd.GeoDataFrame(
#         geometry=ecoscope.base.utils.create_meshgrid(
#             AOI,
#             in_crs=AOI.crs,
#             out_crs=4326,
#             xlen=5000,
#             ylen=5000,
#             return_intersecting_only=False))


# Create grid covering the data extent
# grid_cells = create_meshgrid(points.geometry.unary_union, pixel_size)
# grid = gpd.GeoDataFrame(geometry=grid_cells, crs=points.crs)

# return points, lines, grid

# def test_create_meshgrid(sample_data):
#     points, lines, grid, pixel_size = sample_data
#     # Test if the grid covers the correct extent
#     bounds = points.geometry.unary_union.bounds
#     grid_bounds = grid.total_bounds
#     assert np.allclose(bounds, grid_bounds)


# def test_calculate_density_points(sample_data):
#     points, lines, grid, pixel_size = sample_data
#     grid_with_density = ecoscope.analysis.calculate_feature_density(points, grid, geometry_type="point")
#     # calculate_density(points, grid, geometry_type='point')
#     # Check if the density values are correct
#     expected_density = [1, 1, 1, 1]
#     actual_density = grid_with_density['density'].dropna().values
#     np.testing.assert_array_equal(expected_density, actual_density)

# def test_calculate_density_lines(sample_data):
#     points, lines, grid, pixel_size = sample_data
#     grid_with_density = calculate_density(lines, grid, geometry_type='line')
#     # Check if the density values are correct
#     expected_density = [1.4142135623730951, 1.4142135623730951]  # Lengths of LineStrings divided by 1000
#     actual_density = grid_with_density['density'].dropna().values
#     np.testing.assert_array_almost_equal(expected_density, actual_density)

# def test_grid_to_raster(sample_data, tmpdir):
#     points, lines, grid, pixel_size = sample_data
#     grid_with_density = calculate_density(points, grid, geometry_type='point')
#     grid_crs = grid.crs
#     out_dir = tmpdir
#     rast_name = 'test_density.tif'

#     grid_to_raster(grid_with_density, 'density', out_dir, rast_name, pixel_size, grid_crs)

#     # Check if the raster file is created and has correct dimensions
#     import rasterio
#     with rasterio.open(os.path.join(out_dir, rast_name)) as src:
#         assert src.width == len(np.arange(grid.total_bounds[0], grid.total_bounds[2], pixel_size))
#         assert src.height == len(np.arange(grid.total_bounds[1], grid.total_bounds[3], pixel_size))
#         assert src.count == 1
#         raster_data = src.read(1)
#         assert np.any(~np.isnan(raster_data))

# if __name__ == '__main__':
#     pytest.main()

# import os
# from tempfile import NamedTemporaryFile

# import geopandas as gpd
# import geopandas.testing
# import numpy as np

# import ecoscope


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
