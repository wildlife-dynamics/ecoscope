import os
from tempfile import NamedTemporaryFile
from unittest.mock import patch

import geopandas as gpd
import geopandas.testing
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Point

import ecoscope
from ecoscope.analysis.percentile import get_percentile_area
from ecoscope.analysis.UD import calculate_etd_range, grid_size_from_geographic_extent


@pytest.fixture
def sample_observations():
    gdf = gpd.GeoDataFrame.from_file("tests/sample_data/vector/observations.geojson")
    return gdf


@pytest.fixture(scope="module")
def movebank_trajectory(movebank_gdf):
    relocs = ecoscope.Relocations.from_gdf(
        movebank_gdf,
        groupby_col="individual-local-identifier",
        time_col="timestamp",
        uuid_col="event-id",
    )
    pnts_filter = ecoscope.base.RelocsCoordinateFilter(
        min_x=-5,
        max_x=1,
        min_y=12,
        max_y=18,
        filter_point_coords=[[180, 90], [0, 0]],
    )
    relocs.apply_reloc_filter(pnts_filter, inplace=True)
    relocs.remove_filtered(inplace=True)
    # Relocs are subsampled to keep execution speed low.
    # To run against the full trajectory, drop this slice and compare
    # against tests/test_output/etd_percentile_area.feather (the full-trajectory
    # reference) instead of the _subset reference used below.
    relocs.gdf = relocs.gdf.iloc[::20].copy()
    return ecoscope.Trajectory.from_relocations(relocs)


@pytest.fixture(scope="module")
def raster_profile():
    return ecoscope.io.raster.RasterProfile(
        pixel_size=250.0,
        crs="ESRI:102022",
        nodata_value=np.nan,
        band_count=1,  # Albers Africa Equal Area Conic
    )


@pytest.fixture(scope="module")
def etd_raster_data(movebank_trajectory, raster_profile):
    """Run calculate_etd_range once per module, writing a tif and returning both the
    in-memory result and the result read back from the tif. Lets the two ETD tests
    share the ~10s ETD compute while still covering each code path independently."""
    with NamedTemporaryFile(suffix=".tif", delete=False) as f:
        path = f.name
    try:
        in_memory = calculate_etd_range(
            trajectory=movebank_trajectory,
            output_path=path,
            max_speed_kmhr=1.05 * movebank_trajectory.gdf.speed_kmhr.max(),
            raster_profile=raster_profile,
            expansion_factor=1.3,
        )
        from_file = ecoscope.io.raster.RasterData.from_raster_file(path)
        yield in_memory, from_file
    finally:
        os.unlink(path)


@pytest.fixture
def synthetic_traj():
    timestamps = pd.date_range("2020-01-01", periods=15, freq="1h", tz="UTC")
    rng = np.random.default_rng(seed=0)
    steps = rng.uniform(0.002, 0.008, size=(15, 2))
    coords = np.cumsum(steps, axis=0).tolist()
    gdf = gpd.GeoDataFrame(
        {
            "id": [f"p{i}" for i in range(15)],
            "subject": ["s1"] * 15,
            "fixtime": timestamps,
            "geometry": [Point(x, y) for x, y in coords],
        },
        crs=4326,
    )
    relocs = ecoscope.Relocations.from_gdf(gdf, groupby_col="subject", uuid_col="id")
    return ecoscope.Trajectory.from_relocations(relocs)


def test_calculate_etd_range_skips_write_when_no_output_path(synthetic_traj, raster_profile):
    with patch("ecoscope.analysis.UD.etd_range.raster.RasterPy.write") as mock_write:
        result = calculate_etd_range(
            trajectory=synthetic_traj,
            output_path=None,
            max_speed_kmhr=1.05 * synthetic_traj.gdf.speed_kmhr.max(),
            raster_profile=raster_profile,
            expansion_factor=1.3,
        )
    mock_write.assert_not_called()
    assert isinstance(result, ecoscope.io.raster.RasterData)
    assert result.data.size > 0


def test_calculate_etd_range_writes_when_output_path(synthetic_traj, raster_profile, tmp_path):
    with patch("ecoscope.analysis.UD.etd_range.raster.RasterPy.write") as mock_write:
        calculate_etd_range(
            trajectory=synthetic_traj,
            output_path=str(tmp_path / "out.tif"),
            max_speed_kmhr=1.05 * synthetic_traj.gdf.speed_kmhr.max(),
            raster_profile=raster_profile,
            expansion_factor=1.3,
        )
    mock_write.assert_called_once()


def test_etd_range_percentile_area(etd_raster_data):
    in_memory, _ = etd_raster_data
    percentile_area = get_percentile_area(
        percentile_levels=[99.9], raster_data=in_memory, subject_id="Salif_Keita"
    ).to_crs(4326)

    expected_percentile_area = gpd.read_feather("tests/test_output/etd_percentile_area_subset.feather")
    assert gpd.testing.geom_almost_equals(percentile_area, expected_percentile_area)


def test_etd_range_tif_roundtrip(etd_raster_data):
    in_memory, from_file = etd_raster_data
    in_memory_area = get_percentile_area(
        percentile_levels=[99.9], raster_data=in_memory, subject_id="Salif_Keita"
    ).to_crs(4326)
    from_file_area = get_percentile_area(
        percentile_levels=[99.9], raster_data=from_file, subject_id="Salif_Keita"
    ).to_crs(4326)

    assert gpd.testing.geom_almost_equals(from_file_area, in_memory_area)


def test_reduce_regions(aoi_gdf):
    raster_names = ["tests/sample_data/raster/mara_dem.tif"]
    result = ecoscope.io.raster.reduce_region(aoi_gdf, raster_names, np.mean)
    assert result[raster_names[0]].sum() > 0


def test_grid_size_from_geographic_extent(movebank_relocations, aoi_gdf, sample_observations):
    small_extent = gpd.GeoDataFrame(geometry=[Point(0.0001, 0.0002), Point(0.0002, 0.0001)], crs="EPSG:4326")
    assert 1 == grid_size_from_geographic_extent(small_extent)

    relocs_gdf = movebank_relocations.gdf
    # aoi_gdf.total_bounds = [34.798, -1.901, 36.001, -0.997], smallest extent
    aoi_gdf_cell_size = grid_size_from_geographic_extent(aoi_gdf)
    # sample_observations.total_bounds = array([20.303, -2.197, 39.375,  2.548])
    sample_observations_cell_size = grid_size_from_geographic_extent(sample_observations)
    # Relocs.total_bounds = [-3.099, 0.535, 37.631, 15.736], largest extent
    relocs_cell_size = grid_size_from_geographic_extent(relocs_gdf)

    assert aoi_gdf_cell_size < sample_observations_cell_size < relocs_cell_size
