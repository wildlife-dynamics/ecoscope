import os
from tempfile import NamedTemporaryFile
from unittest.mock import patch

import geopandas as gpd
import geopandas.testing
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import LineString, Point

import ecoscope
from ecoscope.analysis.percentile import get_percentile_area
from ecoscope.analysis.UD import (
    calculate_bbmm_range,
    calculate_etd_range,
    calculate_mcp_range,
    grid_size_from_geographic_extent,
)
from ecoscope.analysis.UD.bbmm_range import estimate_motion_variance


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


def test_calculate_mcp_range_area_increases_with_percentile(movebank_relocations):
    result = calculate_mcp_range(
        relocations=movebank_relocations,
        percentile_levels=[50.0, 90.0],
        crs="ESRI:102022",
        subject_id="Salif Keita",
    )

    assert list(result.columns) == ["subject_id", "percentile", "actual_percentile", "geometry"]
    assert (result["subject_id"] == "Salif Keita").all()
    assert result.crs.to_string() == "ESRI:102022"

    area_by_percentile = result.set_index("percentile").area
    assert area_by_percentile[90.0] > area_by_percentile[50.0]


def test_calculate_mcp_range_accepts_geodataframe_directly(movebank_relocations):
    from_relocations = calculate_mcp_range(relocations=movebank_relocations, percentile_levels=[90.0])
    from_gdf = calculate_mcp_range(relocations=movebank_relocations.gdf, percentile_levels=[90.0])

    geopandas.testing.assert_geodataframe_equal(from_relocations, from_gdf)


def test_calculate_mcp_range_skips_percentile_with_too_few_fixes(movebank_relocations):
    tiny_gdf = movebank_relocations.gdf.iloc[:5].copy()

    result = calculate_mcp_range(relocations=tiny_gdf, percentile_levels=[90.0, 10.0])

    # 10% of 5 fixes rounds down to 0, well under the 3-fix minimum for a hull.
    assert list(result["percentile"]) == [90.0]


def test_calculate_bbmm_range_returns_normalized_raster(synthetic_traj):
    result = calculate_bbmm_range(synthetic_traj.gdf, crs="EPSG:3857")

    assert isinstance(result, ecoscope.io.raster.RasterData)
    assert result.data.shape[0] > 0 and result.data.shape[1] > 0
    assert not np.all(result.data == 0)
    # Normalized to integrate to ~1 over the grid.
    pixel_size = grid_size_from_geographic_extent(synthetic_traj.gdf.to_crs("EPSG:3857"), scale_factor=500)
    assert result.data.sum() * pixel_size * pixel_size == pytest.approx(1.0, abs=0.05)


def test_calculate_bbmm_range_percentile_area_increases_with_percentile(synthetic_traj):
    raster_data = calculate_bbmm_range(synthetic_traj.gdf, crs="EPSG:3857")
    result = get_percentile_area(percentile_levels=[50.0, 90.0], raster_data=raster_data, subject_id="s1")

    area_by_percentile = result.set_index("percentile").area
    assert area_by_percentile[90.0] > area_by_percentile[50.0]


def test_calculate_bbmm_range_excludes_segments_beyond_max_data_gap(synthetic_traj):
    # Every segment in synthetic_traj is a 1-hour (3600s) gap; excluding anything
    # under that drops every segment, leaving an all-zero (not normalized) surface
    # rather than raising a division error.
    result = calculate_bbmm_range(synthetic_traj.gdf, crs="EPSG:3857", max_data_gap_seconds=1000.0)

    assert np.all(result.data == 0)


def test_estimate_motion_variance_returns_positive_value(synthetic_traj):
    sigma_m2 = estimate_motion_variance(synthetic_traj.gdf.to_crs("EPSG:3857"), location_error=20.0)

    assert sigma_m2 > 0


def test_estimate_motion_variance_raises_with_too_few_fixes():
    # A single segment has no interior fixes at all to leave one out from.
    times = pd.date_range("2024-01-01", periods=2, freq="60s", tz="UTC")
    tiny_gdf = gpd.GeoDataFrame(
        [{"geometry": LineString([(0, 0), (10, 10)]), "segment_start": times[0], "segment_end": times[1]}],
        crs="EPSG:3857",
    )

    with pytest.raises(ValueError, match="Not enough interior fixes"):
        estimate_motion_variance(tiny_gdf, location_error=20.0)


def test_calculate_bbmm_range_skips_segment_with_nonpositive_time_lag():
    # Segment index 2's segment_end duplicates its own segment_start, giving
    # it a zero time lag - it should be silently skipped, not raise or corrupt
    # the surface built from the other (valid) segments.
    times = pd.date_range("2024-01-01", periods=6, freq="60s", tz="UTC")
    points = [(0, 0), (10, 10), (20, 20), (20, 20), (30, 30), (40, 40)]
    rows = [
        {
            "geometry": LineString([points[i], points[i + 1]]),
            "segment_start": times[i],
            "segment_end": times[i] if i == 2 else times[i + 1],
        }
        for i in range(5)
    ]
    gdf = gpd.GeoDataFrame(rows, crs="EPSG:3857")

    result = calculate_bbmm_range(gdf, crs="EPSG:3857", location_error=20.0)

    assert result.data.shape[0] > 0 and result.data.shape[1] > 0


def test_calculate_bbmm_range_skips_segment_outside_grid_window(synthetic_traj):
    # `pad` (window_padding_sigma * sigma + pixel_size) always reaches at least
    # one real pixel center in practice, so this branch is effectively
    # unreachable with real data - shrink the grid `_build_grid` returns to an
    # empty one instead, forcing every segment's own local window to miss it.
    import ecoscope.analysis.UD.bbmm_range as bbmm_module

    real_build_grid = bbmm_module._build_grid

    def empty_grid(*args, **kwargs):
        profile, col_centers, row_centers = real_build_grid(*args, **kwargs)
        return profile, col_centers[:0], row_centers[:0]

    with patch.object(bbmm_module, "_build_grid", side_effect=empty_grid):
        result = calculate_bbmm_range(synthetic_traj.gdf, crs="EPSG:3857")

    # Every segment was skipped, so nothing was accumulated onto the grid.
    assert np.all(result.data == 0)
