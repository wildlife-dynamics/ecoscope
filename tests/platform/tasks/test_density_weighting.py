import math

import geopandas as gpd  # type: ignore[import-untyped]
import numpy as np
from shapely.geometry import box

from ecoscope.platform.tasks.analysis import (
    get_density_colormap,
    get_density_legend_title,
    get_weighting_column,
    normalize_density_units,
    set_patrol_weighting_spec,
)
from ecoscope.platform.tasks.analysis._density_weighting import (
    SumWeightingSpec,
    UDWeightingSpec,
    labeled_weighting,
)
from ecoscope.platform.tasks.analysis._patrol_density import (
    PATROL_WEIGHTING_SPECS,
    PatrolWeightingSelection,
)
from ecoscope.platform.tasks.transformation._unit import Unit


def test_set_patrol_weighting_spec():
    assert set_patrol_weighting_spec() is PATROL_WEIGHTING_SPECS["timespan_seconds"]
    for choice in ("timespan_seconds", "dist_meters", "normalised_ltd"):
        weighting = PatrolWeightingSelection(density_sum_column=choice)
        assert set_patrol_weighting_spec(weighting=weighting) is PATROL_WEIGHTING_SPECS[choice]


def test_weighting_selection_schema_shape():
    # percentiles is only revealed by the Normalised (LTD) conditional branch
    schema = PatrolWeightingSelection.model_json_schema()
    assert list(schema["properties"]) == ["density_sum_column"]
    assert "additionalProperties" not in schema
    (branch,) = schema["allOf"]
    assert branch["if"]["properties"]["density_sum_column"]["const"] == "normalised_ltd"
    percentiles = branch["then"]["properties"]["percentiles"]
    # no default on the branch — the pre-fill rides on the param-level default
    assert "default" not in percentiles
    assert "minItems" not in percentiles
    assert percentiles["uniqueItems"] is True


def test_weighting_param_default_seeds_ltd_percentiles():
    from wt_registry.jsonschema import jsonschema_from_task_func

    schema = jsonschema_from_task_func(set_patrol_weighting_spec)
    assert schema["properties"]["weighting"]["default"] == {
        "density_sum_column": "timespan_seconds",
        "percentiles": ["50", "60", "70", "80", "90", "100"],
    }


def test_weighting_selection_clears_orphaned_percentiles():
    weighting = PatrolWeightingSelection(density_sum_column="timespan_seconds", percentiles=["50", "90"])
    assert weighting.percentiles is None


def test_set_patrol_weighting_spec_custom_percentiles():
    weighting = PatrolWeightingSelection(density_sum_column="normalised_ltd", percentiles=["50", "90", "100"])
    spec = set_patrol_weighting_spec(weighting=weighting)
    assert isinstance(spec, UDWeightingSpec)
    assert spec.percentiles == (50.0, 90.0, 100.0)
    # the shared static spec stays untouched
    assert PATROL_WEIGHTING_SPECS["normalised_ltd"].percentiles is None


def test_sum_specs_unchanged():
    time_spec = PATROL_WEIGHTING_SPECS["timespan_seconds"]
    assert isinstance(time_spec, SumWeightingSpec)
    assert (
        time_spec.density_sum_column,
        time_spec.original_unit,
        time_spec.display_unit,
    ) == (
        "timespan_seconds",
        Unit.SECOND,
        Unit.HOUR,
    )
    assert time_spec.option_label == "Time"
    dist_spec = PATROL_WEIGHTING_SPECS["dist_meters"]
    assert isinstance(dist_spec, SumWeightingSpec)
    assert (
        dist_spec.density_sum_column,
        dist_spec.original_unit,
        dist_spec.display_unit,
    ) == (
        "dist_meters",
        Unit.METER,
        Unit.KILOMETER,
    )
    assert dist_spec.option_label == "Distance"


def test_ud_spec():
    ud_spec = PATROL_WEIGHTING_SPECS["normalised_ltd"]
    assert isinstance(ud_spec, UDWeightingSpec)
    assert (ud_spec.option_label, ud_spec.legend_label) == (
        "Normalised (LTD)",
        "Time Spent",
    )
    assert ud_spec.display_unit == Unit.PERCENT
    assert ud_spec.percentiles is None


def test_get_weighting_column():
    assert get_weighting_column(weighting_spec=PATROL_WEIGHTING_SPECS["timespan_seconds"]) == "timespan_seconds"
    assert get_weighting_column(weighting_spec=PATROL_WEIGHTING_SPECS["dist_meters"]) == "dist_meters"


def test_normalize_density_units_time():
    grid = gpd.GeoDataFrame(
        data={"density": [7200.0, 1800.0, np.nan]},
        geometry=[box(0, 0, 1, 1), box(1, 0, 2, 1), box(2, 0, 3, 1)],
        crs="EPSG:3857",
    )
    result = normalize_density_units(df=grid, weighting_spec=PATROL_WEIGHTING_SPECS["timespan_seconds"])
    assert math.isclose(result["density"][0], 2.0)
    assert math.isclose(result["density"][1], 0.5)
    assert np.isnan(result["density"][2])


def test_normalize_density_units_distance():
    grid = gpd.GeoDataFrame(
        data={"density": [2500.0, 750.0]},
        geometry=[box(0, 0, 1, 1), box(1, 0, 2, 1)],
        crs="EPSG:3857",
    )
    result = normalize_density_units(df=grid, weighting_spec=PATROL_WEIGHTING_SPECS["dist_meters"])
    assert math.isclose(result["density"][0], 2.5)
    assert math.isclose(result["density"][1], 0.75)


def test_get_density_legend_title():
    assert get_density_legend_title(weighting_spec=PATROL_WEIGHTING_SPECS["timespan_seconds"]) == "Time (h)"
    assert get_density_legend_title(weighting_spec=PATROL_WEIGHTING_SPECS["dist_meters"]) == "Distance (km)"
    assert get_density_legend_title(weighting_spec=PATROL_WEIGHTING_SPECS["normalised_ltd"]) == "Time Spent (%)"


def test_get_density_colormap():
    assert get_density_colormap(weighting_spec=PATROL_WEIGHTING_SPECS["timespan_seconds"]) == "RdYlGn_r"
    assert get_density_colormap(weighting_spec=PATROL_WEIGHTING_SPECS["dist_meters"]) == "RdYlGn_r"
    assert get_density_colormap(weighting_spec=PATROL_WEIGHTING_SPECS["normalised_ltd"]) == "RdYlGn"


def test_labeled_weighting_replaces_enum_with_labeled_options():
    schema = {"enum": ["timespan_seconds", "dist_meters", "normalised_ltd"]}
    labeled_weighting(PATROL_WEIGHTING_SPECS)(schema)
    assert "enum" not in schema
    assert schema["oneOf"] == [
        {"const": "timespan_seconds", "title": "Time"},
        {"const": "dist_meters", "title": "Distance"},
        {"const": "normalised_ltd", "title": "Normalised (LTD)"},
    ]
