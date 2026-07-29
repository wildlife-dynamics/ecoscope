import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Point

from ecoscope.platform.tasks.config import (
    EtdMethodArgs,
    McpMethodArgs,
    any_is_mcp_method_args,
    call_home_range_from_args,
    get_etd_raster_params_from_args,
    get_home_range_opacity,
    get_opacity_from_combined_params,
    get_stroked_from_args,
    relocations_for_points_overlay,
    set_density_grid_options,
    set_home_range_args,
)
from ecoscope.platform.tasks.config._meta_tasks import DensityGridOptions
from wt_task import task


def test_set_density_grid_options_defaults():
    opts = set_density_grid_options()
    assert isinstance(opts, DensityGridOptions)
    assert opts.opacity == 0.7
    assert opts.crs == "EPSG:3857"
    assert opts.intersecting_only is False
    assert opts.auto_scale_or_custom_cell_size is None


def test_density_grid_options_get_meshgrid_params():
    opts = set_density_grid_options(opacity=0.4, crs="EPSG:6933", intersecting_only=True)
    assert opts.get_meshgrid_params() == {
        "auto_scale_or_custom_cell_size": None,
        "crs": "EPSG:6933",
        "intersecting_only": True,
    }


def test_get_opacity_from_density_grid_options():
    opts = set_density_grid_options(opacity=0.25)
    assert get_opacity_from_combined_params(opts) == 0.25


@pytest.fixture(scope="module")
def synthetic_traj_and_relocs():
    from ecoscope.relocations import Relocations
    from ecoscope.trajectory import Trajectory

    rng = np.random.default_rng(0)
    n = 300
    timestamps = pd.date_range("2020-01-01", periods=n, freq="1h", tz="UTC")
    xy = np.cumsum(rng.uniform(-500, 500, size=(n, 2)), axis=0)
    gdf = gpd.GeoDataFrame(
        {"groupby_col": "s1", "fixtime": timestamps, "geometry": [Point(x, y) for x, y in xy]},
        crs="ESRI:102022",
    )
    traj = Trajectory.from_relocations(Relocations.from_gdf(gdf, groupby_col="groupby_col", time_col="fixtime"))
    relocs_gdf = Trajectory(gdf=traj.gdf).to_relocations().gdf
    return traj.gdf, relocs_gdf


def test_set_home_range_args_defaults_to_mcp():
    """set_home_range_args has no default at the Python level (required, since a raw
    pydantic BaseModel instance would be an unsafe mutable default) - its schema-level
    default (used by RJSF) is MCP-shaped; task(...).validate().call() resolves that."""
    args = task(set_home_range_args).validate().call()
    assert isinstance(args, McpMethodArgs)


def test_etd_mcp_disambiguation_without_shared_discriminator():
    """No `method`/`type` tag field on either model - EtdMethodArgs vs McpMethodArgs is
    told apart structurally: max_speed_factor is required (no default) on EtdMethodArgs,
    so MCP-shaped data (missing that key) fails EtdMethodArgs validation and falls
    through to McpMethodArgs; extra='forbid' on both stops the reverse ambiguity."""
    etd = EtdMethodArgs(opacity=0.7, crs="EPSG:3857", max_speed_factor=1.05, expansion_factor=1.3, percentiles=[50.0])
    mcp = McpMethodArgs(opacity=0.6, crs="ESRI:102022", percentiles=[50.0, 90.0])
    assert isinstance(etd, EtdMethodArgs)
    assert isinstance(mcp, McpMethodArgs)


def test_get_stroked_from_args():
    etd = EtdMethodArgs(opacity=0.7, crs="EPSG:3857", max_speed_factor=1.05, expansion_factor=1.3, percentiles=[50.0])
    mcp = McpMethodArgs(opacity=0.6, crs="ESRI:102022", percentiles=[50.0])
    assert get_stroked_from_args(etd) is False
    assert get_stroked_from_args(mcp) is True


def test_relocations_for_points_overlay(synthetic_traj_and_relocs):
    _, relocs_gdf = synthetic_traj_and_relocs
    etd = EtdMethodArgs(opacity=0.7, crs="EPSG:3857", max_speed_factor=1.05, expansion_factor=1.3, percentiles=[50.0])
    mcp = McpMethodArgs(opacity=0.6, crs="ESRI:102022", percentiles=[50.0])

    assert len(relocations_for_points_overlay(relocs_gdf, mcp)) == len(relocs_gdf)
    assert len(relocations_for_points_overlay(relocs_gdf, etd)) == 0


def test_get_home_range_opacity():
    etd = EtdMethodArgs(opacity=0.7, crs="EPSG:3857", max_speed_factor=1.05, expansion_factor=1.3, percentiles=[50.0])
    mcp = McpMethodArgs(opacity=0.4, crs="ESRI:102022", percentiles=[50.0])
    assert get_home_range_opacity(etd) == 0.7
    assert get_home_range_opacity(mcp) == 0.4


def test_get_etd_raster_params_from_args_passes_through_etd():
    etd = EtdMethodArgs(opacity=0.7, crs="EPSG:3857", max_speed_factor=2.0, expansion_factor=1.5, percentiles=[50.0])
    raster_params = get_etd_raster_params_from_args(etd)
    assert raster_params.crs == "EPSG:3857"
    assert raster_params.max_speed_factor == 2.0
    assert raster_params.expansion_factor == 1.5


def test_any_is_mcp_method_args():
    """skipif condition used to skip the ETD-only raster entirely when MCP is selected."""
    etd = EtdMethodArgs(opacity=0.7, crs="EPSG:3857", max_speed_factor=1.05, expansion_factor=1.3, percentiles=[50.0])
    mcp = McpMethodArgs(opacity=0.6, crs="ESRI:102022", percentiles=[50.0])
    assert any_is_mcp_method_args(etd) is False
    assert any_is_mcp_method_args(mcp) is True


def test_get_etd_raster_params_from_args_asserts_when_mcp_selected():
    """Should never actually be called with MCP args - the spec.yaml task instance is
    skipped entirely via any_is_mcp_method_args in that case."""
    mcp = McpMethodArgs(opacity=0.6, crs="ESRI:102022", percentiles=[50.0])
    with pytest.raises(AssertionError):
        get_etd_raster_params_from_args(mcp)


def test_call_home_range_from_args_etd_and_mcp_share_schema(synthetic_traj_and_relocs):
    traj_gdf, _ = synthetic_traj_and_relocs
    etd = EtdMethodArgs(opacity=0.7, crs="EPSG:3857", max_speed_factor=1.05, expansion_factor=1.3, percentiles=[50.0])
    mcp = McpMethodArgs(opacity=0.6, crs="ESRI:102022", percentiles=[50.0, 90.0])

    etd_result = call_home_range_from_args(traj_gdf, etd)
    mcp_result = call_home_range_from_args(traj_gdf, mcp)

    assert set(etd_result.columns) == set(mcp_result.columns) == {"percentile", "geometry", "area_sqkm"}
