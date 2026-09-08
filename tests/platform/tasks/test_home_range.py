from importlib.resources import files

import geopandas as gpd  # type: ignore[import-untyped]
import pytest
from shapely.geometry import Point  # type: ignore[import-untyped]

from ecoscope.platform.tasks.config._home_range import (
    BbmmLocationErrorSettings,
    BbmmMethodArgs,
    BbmmRasterArgs,
    EtdMethodArgs,
    EtdSpeedSettings,
    McpMethodArgs,
    McpShowRelocationsSettings,
    any_is_non_bbmm_method_args,
    any_is_non_etd_method_args,
    call_home_range_from_args,
    get_bbmm_raster_params_from_args,
    get_etd_raster_params_from_args,
    get_rings_correction_from_args,
    get_stroked_from_args,
    relocations_for_points_overlay,
    set_home_range_args,
)


@pytest.fixture(scope="module")
def trajectory_gdf():
    example_input_df_path = (
        files("ecoscope.platform.tasks.preprocessing") / "relocations-to-trajectory.example-return.parquet"
    )
    gdf = gpd.read_parquet(example_input_df_path)
    # Subsampled to keep this test's runtime low.
    return gdf.iloc[::10].copy()


@pytest.fixture
def etd_args():
    return EtdMethodArgs(
        crs="EPSG:3857",
        speed_settings=EtdSpeedSettings(max_speed_factor=1.05),
        percentiles=[50.0, 90.0],
    )


@pytest.fixture
def mcp_args():
    return McpMethodArgs(crs="EPSG:3857", percentiles=[50.0, 90.0])


@pytest.fixture
def bbmm_args():
    return BbmmMethodArgs(
        crs="EPSG:3857",
        location_error_settings=BbmmLocationErrorSettings(location_error=20.0),
        percentiles=[50.0, 90.0],
    )


def test_etd_method_args_get_etd_params(etd_args):
    assert etd_args.get_etd_params() == {
        "auto_scale_or_custom_cell_size": None,
        "crs": "EPSG:3857",
        "max_speed_factor": 1.05,
        "expansion_factor": 1.3,
        "percentiles": [50.0, 90.0],
    }


def test_mcp_method_args_get_mcp_params(mcp_args):
    assert mcp_args.get_mcp_params() == {"crs": "EPSG:3857", "percentiles": [50.0, 90.0]}


def test_bbmm_method_args_get_bbmm_params(bbmm_args):
    assert bbmm_args.get_bbmm_params() == {
        "crs": "EPSG:3857",
        "location_error": 20.0,
        "time_step_seconds": 60.0,
        "expansion_factor": 1.3,
        "percentiles": [50.0, 90.0],
        "max_data_gap_seconds": 14400.0,
    }


def test_bbmm_raster_args_get_bbmm_params():
    args = BbmmRasterArgs(
        crs="EPSG:3857",
        location_error=20.0,
        time_step_seconds=60.0,
        expansion_factor=1.3,
        max_data_gap_seconds=14400.0,
    )
    assert args.get_bbmm_params() == {
        "crs": "EPSG:3857",
        "location_error": 20.0,
        "time_step_seconds": 60.0,
        "expansion_factor": 1.3,
        "max_data_gap_seconds": 14400.0,
    }


def test_set_home_range_args_passthrough(mcp_args):
    assert set_home_range_args(mcp_args) is mcp_args


def test_call_home_range_from_args_etd(trajectory_gdf, etd_args):
    result = call_home_range_from_args(trajectory_gdf, etd_args)
    assert list(result.columns) == ["percentile", "geometry", "area_sqkm"]
    assert len(result) == 2


def test_call_home_range_from_args_bbmm(trajectory_gdf, bbmm_args):
    result = call_home_range_from_args(trajectory_gdf, bbmm_args)
    assert list(result.columns) == ["percentile", "geometry", "area_sqkm"]
    assert len(result) == 2


def test_call_home_range_from_args_mcp(trajectory_gdf, mcp_args):
    result = call_home_range_from_args(trajectory_gdf, mcp_args)
    assert list(result.columns) == ["percentile", "geometry", "area_sqkm"]
    assert len(result) == 2


def test_get_stroked_from_args(etd_args, mcp_args, bbmm_args):
    assert get_stroked_from_args(mcp_args) is True
    assert get_stroked_from_args(etd_args) is False
    assert get_stroked_from_args(bbmm_args) is False


def test_get_rings_correction_from_args(etd_args, mcp_args, bbmm_args):
    assert get_rings_correction_from_args(mcp_args) is True
    assert get_rings_correction_from_args(etd_args) is False
    assert get_rings_correction_from_args(bbmm_args) is False


@pytest.fixture
def relocations_gdf():
    return gpd.GeoDataFrame(
        {"fixtime": ["2020-01-01T00:00:00Z", "2020-01-02T00:00:00Z"]},
        geometry=[Point(0, 0), Point(1, 1)],
        crs="EPSG:4326",
    )


def test_relocations_for_points_overlay_mcp_shown(relocations_gdf):
    args = McpMethodArgs(
        crs="EPSG:3857",
        percentiles=[50.0],
        show_relocations_settings=McpShowRelocationsSettings(show_relocations=True),
    )
    result = relocations_for_points_overlay(relocations_gdf, args)
    assert len(result) == 2


def test_relocations_for_points_overlay_mcp_hidden_by_default(relocations_gdf, mcp_args):
    result = relocations_for_points_overlay(relocations_gdf, mcp_args)
    assert len(result) == 0


def test_relocations_for_points_overlay_etd_always_empty(relocations_gdf, etd_args):
    result = relocations_for_points_overlay(relocations_gdf, etd_args)
    assert len(result) == 0


def test_any_is_non_etd_method_args(etd_args, mcp_args):
    assert any_is_non_etd_method_args(etd_args, etd_args) is False
    assert any_is_non_etd_method_args(etd_args, mcp_args) is True


def test_get_etd_raster_params_from_args(etd_args):
    params = get_etd_raster_params_from_args(etd_args)
    assert params.crs == "EPSG:3857"
    assert params.max_speed_factor == 1.05
    assert params.expansion_factor == 1.3
    assert params.percentiles == [50.0, 90.0]
    assert params.opacity == 1.0
    assert params.band_count == 1
    assert params.nodata_value == "nan"


def test_any_is_non_bbmm_method_args(bbmm_args, mcp_args):
    assert any_is_non_bbmm_method_args(bbmm_args, bbmm_args) is False
    assert any_is_non_bbmm_method_args(bbmm_args, mcp_args) is True


def test_get_bbmm_raster_params_from_args(bbmm_args):
    params = get_bbmm_raster_params_from_args(bbmm_args)
    assert params.crs == "EPSG:3857"
    assert params.location_error == 20.0
    assert params.time_step_seconds == 60.0
    assert params.expansion_factor == 1.3
    assert params.max_data_gap_seconds == 14400.0
