from importlib.resources import files

import geopandas as gpd  # type: ignore[import-untyped]
import numpy as np
import pytest
import rasterio as rio  # type: ignore[import-untyped]

import ecoscope.analysis.UD as UD  # type: ignore[import-untyped]
from ecoscope.io.raster import RasterData
from ecoscope.platform.tasks.analysis._raster import (
    BbmmRasterArgs,
    _build_output_path,
    _filename_prefix_from_group_key,
    _hash_grouper_key,
    _remove_file_scheme,
    export_geotiff,
    generate_bbmm_raster,
    generate_etd_raster,
)
from ecoscope.platform.tasks.analysis._time_density import AutoScaleGridCellSize, CustomGridCellSize
from ecoscope.platform.tasks.config._meta_tasks import EtdArgsWithOpacity


@pytest.fixture(scope="module")
def trajectory_gdf():
    example_input_df_path = (
        files("ecoscope.platform.tasks.preprocessing") / "relocations-to-trajectory.example-return.parquet"
    )
    gdf = gpd.read_parquet(example_input_df_path)
    # Subsampled to keep this test's runtime low.
    return gdf.iloc[::10].copy()


def test_hash_grouper_key_is_deterministic():
    group_key = (("subject_name", "=", "eco1"),)
    assert _hash_grouper_key(group_key) == _hash_grouper_key(group_key)
    assert len(_hash_grouper_key(group_key)) == 6


def test_hash_grouper_key_order_independent():
    a = (("subject_name", "=", "eco1"), ("month", "=", "01"))
    b = (("month", "=", "01"), ("subject_name", "=", "eco1"))
    assert _hash_grouper_key(a) == _hash_grouper_key(b)


def test_filename_prefix_from_group_key_none():
    assert _filename_prefix_from_group_key(None) is None
    assert _filename_prefix_from_group_key(()) is None


def test_filename_prefix_from_group_key_present():
    group_key = (("subject_name", "=", "eco1"),)
    assert _filename_prefix_from_group_key(group_key) == _hash_grouper_key(group_key)


@pytest.mark.parametrize(
    "path,expected",
    [
        ("/plain/path", "/plain/path"),
        ("file:///abs/path", "/abs/path"),
        ("file://localhost/abs/path", "/abs/path"),
        # no path component - urlparse puts the content in netloc instead
        ("file://onlyhost", "onlyhost"),
    ],
)
def test_remove_file_scheme(path, expected):
    assert _remove_file_scheme(path) == expected


def test_remove_file_scheme_windows(monkeypatch):
    monkeypatch.setattr("os.name", "nt")
    assert _remove_file_scheme("file:///C:/Users/Admin/file.tif") == "C:\\Users\\Admin\\file.tif"


def test_build_output_path_no_group_key(tmp_path):
    output_path = _build_output_path(str(tmp_path), "etd_raster", None)
    assert output_path == str(tmp_path / "etd_raster.tif")


def test_build_output_path_with_group_key(tmp_path):
    group_key = (("subject_name", "=", "eco1"),)
    output_path = _build_output_path(str(tmp_path), "etd_raster", group_key)
    prefix = _hash_grouper_key(group_key)
    assert output_path == str(tmp_path / f"{prefix}_etd_raster.tif")


def test_build_output_path_creates_dir(tmp_path):
    output_dir = tmp_path / "nested" / "dir"
    _build_output_path(str(output_dir), "etd_raster", None)
    assert output_dir.is_dir()


def test_export_geotiff_writes_file_with_nan_nodata(tmp_path):
    data = np.array([[1.0, 0.0], [np.nan, 2.0]], dtype="float32")
    raster_data = RasterData(data=data, crs="EPSG:3857", transform=rio.Affine.identity())
    output_path = str(tmp_path / "out.tif")

    export_geotiff(raster_data, output_path)

    with rio.open(output_path) as src:
        assert src.count == 1
        arr = src.read(1)
        assert np.isnan(src.nodata)
        # both the explicit nan and the exact-zero cell are masked to nodata
        assert np.isnan(arr[1, 0])
        assert np.isnan(arr[0, 1])
        assert arr[0, 0] == 1.0
        assert arr[1, 1] == 2.0


def test_export_geotiff_numeric_nodata(tmp_path):
    data = np.array([[1.0, -9999.0]], dtype="float32")
    raster_data = RasterData(data=data, crs="EPSG:3857", transform=rio.Affine.identity())
    output_path = str(tmp_path / "out.tif")

    export_geotiff(raster_data, output_path, nodata=-9999.0)

    with rio.open(output_path) as src:
        assert src.nodata == -9999.0


def test_generate_etd_raster_auto_scale(trajectory_gdf, tmp_path):
    combined_params = EtdArgsWithOpacity(
        opacity=1.0,
        auto_scale_or_custom_cell_size=None,
        crs="ESRI:102022",
        nodata_value="nan",
        band_count=1,
        max_speed_factor=1.05,
        expansion_factor=1.3,
        percentiles=[50.0, 90.0],
    )

    output_path = generate_etd_raster(trajectory_gdf, combined_params, output_dir=str(tmp_path))

    assert output_path == str(tmp_path / "etd_raster.tif")
    with rio.open(output_path) as src:
        assert src.count == 1


def test_generate_etd_raster_custom_cell_size(trajectory_gdf, tmp_path):
    combined_params = EtdArgsWithOpacity(
        opacity=1.0,
        auto_scale_or_custom_cell_size=CustomGridCellSize(grid_cell_size=250.0),
        crs="ESRI:102022",
        nodata_value="nan",
        band_count=1,
        max_speed_factor=1.05,
        expansion_factor=1.3,
        percentiles=[50.0],
    )

    output_path = generate_etd_raster(trajectory_gdf, combined_params, output_dir=str(tmp_path))

    with rio.open(output_path) as src:
        assert abs(src.res[0] - 250.0) < 1e-6


def test_generate_etd_raster_custom_cell_size_none_raises(trajectory_gdf, tmp_path):
    # CustomGridCellSize's own gt=0 constraint rejects a validated None, so
    # this uses model_construct to bypass validation and still exercise
    # generate_etd_raster's own defensive check.
    combined_params = EtdArgsWithOpacity(
        opacity=1.0,
        auto_scale_or_custom_cell_size=CustomGridCellSize.model_construct(grid_cell_size=None),
        crs="ESRI:102022",
        nodata_value="nan",
        band_count=1,
        max_speed_factor=1.05,
        expansion_factor=1.3,
        percentiles=[50.0],
    )

    with pytest.raises(ValueError, match="grid_cell_size must be set"):
        generate_etd_raster(trajectory_gdf, combined_params, output_dir=str(tmp_path))


def test_generate_etd_raster_empty_trajectory_raises(trajectory_gdf, tmp_path):
    combined_params = EtdArgsWithOpacity(
        opacity=1.0,
        auto_scale_or_custom_cell_size=AutoScaleGridCellSize(),
        crs="ESRI:102022",
        nodata_value="nan",
        band_count=1,
        max_speed_factor=1.05,
        expansion_factor=1.3,
        percentiles=[50.0],
    )

    with pytest.raises(ValueError, match="`trajectory_gdf` is empty"):
        generate_etd_raster(trajectory_gdf.iloc[0:0], combined_params, output_dir=str(tmp_path))


def test_generate_etd_raster_no_data_generated_raises(trajectory_gdf, tmp_path, monkeypatch):
    # Forcing calculate_etd_range to actually return empty/None data isn't
    # practical with real trajectory data, so mock it directly to exercise
    # this defensive check.
    monkeypatch.setattr(UD, "calculate_etd_range", lambda **kwargs: None)
    combined_params = EtdArgsWithOpacity(
        opacity=1.0,
        auto_scale_or_custom_cell_size=AutoScaleGridCellSize(),
        crs="ESRI:102022",
        nodata_value="nan",
        band_count=1,
        max_speed_factor=1.05,
        expansion_factor=1.3,
        percentiles=[50.0],
    )

    with pytest.raises(ValueError, match="no raster data was generated"):
        generate_etd_raster(trajectory_gdf, combined_params, output_dir=str(tmp_path))


def test_generate_bbmm_raster(trajectory_gdf, tmp_path):
    combined_params = BbmmRasterArgs(
        crs="ESRI:102022",
        location_error=20.0,
        time_step_seconds=60.0,
        expansion_factor=1.3,
        max_data_gap_seconds=14400.0,
    )

    output_path = generate_bbmm_raster(trajectory_gdf, combined_params, output_dir=str(tmp_path))

    assert output_path == str(tmp_path / "bbmm_raster.tif")
    with rio.open(output_path) as src:
        assert src.count == 1


def test_generate_bbmm_raster_empty_trajectory_raises(trajectory_gdf, tmp_path):
    combined_params = BbmmRasterArgs(
        crs="ESRI:102022",
        location_error=20.0,
        time_step_seconds=60.0,
        expansion_factor=1.3,
        max_data_gap_seconds=14400.0,
    )

    with pytest.raises(ValueError, match="`trajectory_gdf` is empty"):
        generate_bbmm_raster(trajectory_gdf.iloc[0:0], combined_params, output_dir=str(tmp_path))


def test_generate_bbmm_raster_no_data_generated_raises(trajectory_gdf, tmp_path, monkeypatch):
    # Same rationale as the ETD version: mock calculate_bbmm_range directly
    # rather than trying to coax real data into producing an empty raster.
    monkeypatch.setattr(UD, "calculate_bbmm_range", lambda *args, **kwargs: None)
    combined_params = BbmmRasterArgs(
        crs="ESRI:102022",
        location_error=20.0,
        time_step_seconds=60.0,
        expansion_factor=1.3,
        max_data_gap_seconds=14400.0,
    )

    with pytest.raises(ValueError, match="no raster data was generated"):
        generate_bbmm_raster(trajectory_gdf, combined_params, output_dir=str(tmp_path))


def test_generate_bbmm_raster_with_group_key(trajectory_gdf, tmp_path):
    combined_params = BbmmRasterArgs(
        crs="ESRI:102022",
        location_error=20.0,
        time_step_seconds=60.0,
        expansion_factor=1.3,
        max_data_gap_seconds=14400.0,
    )
    group_key = (("subject_name", "=", "eco1"),)

    output_path = generate_bbmm_raster(trajectory_gdf, combined_params, output_dir=str(tmp_path), group_key=group_key)

    prefix = _hash_grouper_key(group_key)
    assert output_path == str(tmp_path / f"{prefix}_bbmm_raster.tif")
