import importlib

import geopandas as gpd  # type: ignore[import-untyped]
import pytest
from shapely.geometry import box
from wt_contracts import declared_structure
from wt_task.cache import resolve_structure

from ecoscope.platform.tasks.analysis._patrol_summary import CountAggregation, SumOfColumnAggregation
from ecoscope.platform.tasks.analysis._time_density import CustomGridCellSize
from ecoscope.platform.tasks.config import (
    EncounterRateMapOptions,
    call_meshgrid_from_map_options,
    get_agg_column_from_map_options,
    get_base_maps_from_map_options,
    get_mask_threshold_from_map_options,
    set_encounter_rate_map_options,
)
from ecoscope.platform.tasks.results._map_utils import TileLayer


def test_set_encounter_rate_map_options_defaults():
    opts = set_encounter_rate_map_options()
    assert isinstance(opts, EncounterRateMapOptions)
    assert opts.aggregation == CountAggregation()
    assert opts.threshold_km == 0.2
    assert opts.auto_scale_or_custom_cell_size is None
    assert opts.crs == "EPSG:3857"
    assert opts.intersecting_only is False
    assert [(t.layer_name, t.opacity) for t in opts.base_maps] == [("TERRAIN", 1), ("SATELLITE", 0.5)]


def test_set_encounter_rate_map_options_passes_values_through():
    cell_size = CustomGridCellSize(grid_cell_size=500.0)
    base_maps = [TileLayer(layer_name="ROADMAP")]
    opts = set_encounter_rate_map_options(
        aggregation=SumOfColumnAggregation(column="number_of_animals"),
        threshold_km=1.5,
        auto_scale_or_custom_cell_size=cell_size,
        crs="EPSG:32736",
        intersecting_only=True,
        base_maps=base_maps,
    )
    assert opts.aggregation == SumOfColumnAggregation(column="number_of_animals")
    assert opts.get_meshgrid_params() == {
        "auto_scale_or_custom_cell_size": cell_size,
        "crs": "EPSG:32736",
        "intersecting_only": True,
    }
    assert get_mask_threshold_from_map_options(opts) == 1.5
    assert get_base_maps_from_map_options(opts) == base_maps


def test_get_agg_column_from_map_options():
    assert get_agg_column_from_map_options(set_encounter_rate_map_options()) == ""
    summed = set_encounter_rate_map_options(aggregation=SumOfColumnAggregation(column="number_of_animals"))
    assert get_agg_column_from_map_options(summed) == "number_of_animals"
    # Sum mode before a column is chosen degrades to counting.
    assert get_agg_column_from_map_options(set_encounter_rate_map_options(aggregation=SumOfColumnAggregation())) == ""


def test_call_meshgrid_from_map_options():
    aoi = gpd.GeoDataFrame(geometry=[box(0, 0, 2000, 1000)], crs="EPSG:3857")
    opts = set_encounter_rate_map_options(auto_scale_or_custom_cell_size=CustomGridCellSize(grid_cell_size=500.0))
    grid = call_meshgrid_from_map_options(aoi=aoi, options=opts)
    assert grid.crs == "EPSG:3857"
    assert len(grid) >= 8
    assert grid.geometry.area.round().eq(250_000).all()


def test_encounter_rate_map_options_codec_round_trips():
    tree = resolve_structure(declared_structure(set_encounter_rate_map_options))
    assert tree is not None
    value = set_encounter_rate_map_options(
        aggregation=SumOfColumnAggregation(column="number_of_animals"),
        auto_scale_or_custom_cell_size=CustomGridCellSize(grid_cell_size=500.0),
    )
    back = tree.decode(tree.encode(value))
    assert back == value
    assert tree.digest(back) == tree.digest(value)
    assert tree.digest(value) != tree.digest(set_encounter_rate_map_options())


@pytest.mark.parametrize(
    ("subpackage", "name"),
    [
        ("config", "set_encounter_rate_map_options"),
        ("config", "call_meshgrid_from_map_options"),
        ("config", "get_agg_column_from_map_options"),
        ("config", "get_base_maps_from_map_options"),
        ("config", "get_mask_threshold_from_map_options"),
        ("analysis", "calculate_encounter_rate_grid"),
        ("analysis", "drop_inactive_grid_cells"),
        ("analysis", "mask_low_effort_cells"),
        ("transformation", "merge_two_dataframes"),
    ],
)
def test_encounter_rate_map_tasks_are_registered_and_cacheable(subpackage, name):
    """Upstreamed from ecoscope-workflows-ext-custom under the same names, which workflows reference bare."""
    from wt_registry import get_registry

    package = importlib.import_module(f"ecoscope.platform.tasks.{subpackage}")
    assert name in package.__all__
    (entry,) = [e for e in get_registry().values() if e.function_name == name]
    assert entry.module_path.startswith(package.__name__)
    assert entry.declares_codec, entry.cache_declaration_error
