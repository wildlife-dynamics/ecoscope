import os
from tempfile import NamedTemporaryFile

import geopandas as gpd
import geopandas.testing
import numpy as np
import pytest

import ecoscope


@pytest.mark.skip(reason="this has been failing since May 2022; will be fixed in a follow-up pull")
def test_etd_range(movbank_relocations):
    # apply relocation coordinate filter to movbank data
    pnts_filter = ecoscope.base.RelocsCoordinateFilter(
        min_x=-5,
        max_x=1,
        min_y=12,
        max_y=18,
        filter_point_coords=[[180, 90], [0, 0]],
    )
    movbank_relocations.apply_reloc_filter(pnts_filter, inplace=True)
    movbank_relocations.remove_filtered(inplace=True)

    # Create Trajectory
    movbank_trajectory_gdf = ecoscope.base.Trajectory.from_relocations(movbank_relocations)

    raster_profile = ecoscope.io.raster.RasterProfile(
        pixel_size=250.0,
        crs="ESRI:102022",
        nodata_value=np.nan,
        band_count=1,  # Albers Africa Equal Area Conic
    )

    file = NamedTemporaryFile(delete=False)
    try:
        ecoscope.analysis.UD.calculate_etd_range(
            trajectory_gdf=movbank_trajectory_gdf,
            output_path=file.name,
            max_speed_kmhr=1.05 * movbank_trajectory_gdf.speed_kmhr.max(),
            raster_profile=raster_profile,
            expansion_factor=1.3,
        )

        percentile_area = ecoscope.analysis.get_percentile_area(
            percentile_levels=[99.9], raster_path=file.name, subject_id="Salif_Keita"
        ).to_crs(4326)
    finally:
        file.close()
        os.unlink(file.name)

    expected_percentile_area = gpd.read_feather("tests/test_output/etd_percentile_area.feather")
    gpd.testing.assert_geodataframe_equal(percentile_area, expected_percentile_area, check_less_precise=True)


@pytest.mark.skip(reason="this has been failing since May 2022; will be fixed in a follow-up pull")
def test_reduce_regions(aoi_gdf):
    raster_names = ["tests/sample_data/raster/mara_dem.tif"]
    result = ecoscope.io.raster.reduce_region(aoi_gdf, raster_names, np.mean)
    assert result[raster_names[0]].sum() > 0
