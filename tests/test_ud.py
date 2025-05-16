import os
from tempfile import NamedTemporaryFile

import geopandas as gpd
import geopandas.testing
import numpy as np
import pytest

import ecoscope
from ecoscope.analysis.UD import calculate_etd_range, grid_size_from_geographic_extent
from ecoscope.analysis.percentile import get_percentile_area


@pytest.fixture
def sample_observations():
    gdf = gpd.GeoDataFrame.from_file("tests/sample_data/vector/observations.geojson")
    return gdf


@pytest.fixture
def movebank_trajectory(movebank_relocations):
    # apply relocation coordinate filter to movebank data
    pnts_filter = ecoscope.base.RelocsCoordinateFilter(
        min_x=-5,
        max_x=1,
        min_y=12,
        max_y=18,
        filter_point_coords=[[180, 90], [0, 0]],
    )
    movebank_relocations.apply_reloc_filter(pnts_filter, inplace=True)
    movebank_relocations.remove_filtered(inplace=True)

    # Create Trajectory
    return ecoscope.Trajectory.from_relocations(movebank_relocations)


@pytest.fixture
def raster_profile():
    return ecoscope.io.raster.RasterProfile(
        pixel_size=250.0,
        crs="ESRI:102022",
        nodata_value=np.nan,
        band_count=1,  # Albers Africa Equal Area Conic
    )


def test_etd_range_with_tif_file(movebank_trajectory, raster_profile):
    file = NamedTemporaryFile(delete=False)
    try:
        calculate_etd_range(
            trajectory=movebank_trajectory,
            output_path=file.name,
            max_speed_kmhr=1.05 * movebank_trajectory.gdf.speed_kmhr.max(),
            raster_profile=raster_profile,
            expansion_factor=1.3,
        )

        raster_data = ecoscope.io.raster.RasterData.from_raster_file(file.name)

        percentile_area = get_percentile_area(
            percentile_levels=[99.9], raster_data=raster_data, subject_id="Salif_Keita"
        ).to_crs(4326)
    finally:
        file.close()
        os.unlink(file.name)

    expected_percentile_area = gpd.read_feather("tests/test_output/etd_percentile_area.feather")
    gpd.testing.geom_almost_equals(percentile_area, expected_percentile_area)


def test_etd_range_without_tif_file(movebank_trajectory, raster_profile):
    file = NamedTemporaryFile(delete=False)
    try:
        raster_data = calculate_etd_range(
            trajectory=movebank_trajectory,
            max_speed_kmhr=1.05 * movebank_trajectory.gdf.speed_kmhr.max(),
            raster_profile=raster_profile,
            expansion_factor=1.3,
        )

        percentile_area = get_percentile_area(
            percentile_levels=[99.9], raster_data=raster_data, subject_id="Salif_Keita"
        ).to_crs(4326)
    finally:
        file.close()
        os.unlink(file.name)

    expected_percentile_area = gpd.read_feather("tests/test_output/etd_percentile_area.feather")
    gpd.testing.geom_almost_equals(percentile_area, expected_percentile_area)


def test_reduce_regions(aoi_gdf):
    raster_names = ["tests/sample_data/raster/mara_dem.tif"]
    result = ecoscope.io.raster.reduce_region(aoi_gdf, raster_names, np.mean)
    assert result[raster_names[0]].sum() > 0


def test_grid_size_from_geographic_extent(movebank_relocations, aoi_gdf, sample_observations):
    relocs_gdf = movebank_relocations.gdf
    # aoi_gdf.total_bounds = [34.798, -1.901, 36.001, -0.997], smallest extent
    aoi_gdf_cell_size = grid_size_from_geographic_extent(aoi_gdf)
    # sample_observations.total_bounds = array([20.303, -2.197, 39.375,  2.548])
    sample_observations_cell_size = grid_size_from_geographic_extent(sample_observations)
    # Relocs.total_bounds = [-3.099, 0.535, 37.631, 15.736], largest extent
    relocs_cell_size = grid_size_from_geographic_extent(relocs_gdf)

    assert aoi_gdf_cell_size < sample_observations_cell_size < relocs_cell_size
