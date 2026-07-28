import math
from importlib.resources import files

import geopandas as gpd  # type: ignore[import-untyped]
import numpy as np
import pytest
from shapely.geometry import box

from ecoscope.platform.tasks.analysis import (
    calculate_classified_track_density,
    calculate_feature_density,
    calculate_linear_time_density,
    get_density_legend_title,
    get_weighting_column,
    normalize_density_units,
    set_patrol_weighting_spec,
)
from ecoscope.platform.tasks.analysis._create_meshgrid import create_meshgrid
from ecoscope.platform.tasks.analysis._density_weighting import labeled_weighting
from ecoscope.platform.tasks.analysis._patrol_density import (
    PATROL_WEIGHTING_SPECS,
    PatrolWeightingSelection,
)
from ecoscope.platform.tasks.analysis._time_density import AutoScaleGridCellSize
from ecoscope.platform.tasks.transformation._classification import (
    DefaultLabels,
    SharedArgs,
    apply_classification,
)
from ecoscope.platform.tasks.transformation._filtering import drop_nan_values_by_column
from ecoscope.platform.tasks.transformation._sorting import sort_values
from ecoscope.platform.tasks.transformation._unit import Unit


@pytest.fixture
def trajectory_gdf():
    example_input_df_path = (
        files("ecoscope.platform.tasks.preprocessing") / "relocations-to-trajectory.example-return.parquet"
    )
    return gpd.read_parquet(example_input_df_path)


@pytest.fixture
def meshgrid(trajectory_gdf):
    return create_meshgrid(
        aoi=trajectory_gdf,
        auto_scale_or_custom_cell_size=AutoScaleGridCellSize(),
    )


def test_set_patrol_weighting_spec():
    assert set_patrol_weighting_spec() is PATROL_WEIGHTING_SPECS["timespan_seconds"]
    for choice in ("timespan_seconds", "dist_meters", "normalised_ltd"):
        weighting = PatrolWeightingSelection(density_sum_column=choice)
        assert set_patrol_weighting_spec(weighting=weighting) is PATROL_WEIGHTING_SPECS[choice]


def test_weighting_selection_schema_shape():
    # percentiles is hidden from base properties (SkipJsonSchema) and only
    # revealed by the dependency branch for the Normalised (LTD) selection —
    # the StatSummaryParam convert_units pattern.
    schema = PatrolWeightingSelection.model_json_schema()
    assert list(schema["properties"]) == ["density_sum_column"]
    assert "additionalProperties" not in schema
    branches = schema["dependencies"]["density_sum_column"]["oneOf"]
    ltd_branch = branches[-1]["properties"]
    assert ltd_branch["density_sum_column"]["const"] == "normalised_ltd"
    percentiles = ltd_branch["percentiles"]
    # pre-filled like the patrols workflow's Time Density Map; the seeded
    # orphan for non-LTD selections is cleared by clear_orphaned_percentiles
    assert percentiles["default"] == ["50", "60", "70", "80", "90", "100"]
    assert "minItems" not in percentiles
    assert percentiles["uniqueItems"] is True


def test_weighting_selection_clears_orphaned_percentiles():
    weighting = PatrolWeightingSelection(density_sum_column="timespan_seconds", percentiles=["50", "90"])
    assert weighting.percentiles is None


def test_set_patrol_weighting_spec_custom_percentiles():
    weighting = PatrolWeightingSelection(density_sum_column="normalised_ltd", percentiles=["50", "90", "100"])
    spec = set_patrol_weighting_spec(weighting=weighting)
    assert spec.percentiles == (50.0, 90.0, 100.0)
    assert spec.mode == "ltd"
    # the shared static spec stays untouched
    assert PATROL_WEIGHTING_SPECS["normalised_ltd"].percentiles is None


def test_sum_specs_unchanged():
    time_spec = PATROL_WEIGHTING_SPECS["timespan_seconds"]
    assert (time_spec.density_sum_column, time_spec.original_unit, time_spec.display_unit) == (
        "timespan_seconds",
        Unit.SECOND,
        Unit.HOUR,
    )
    assert (time_spec.option_label, time_spec.mode) == ("Time", "sum")
    dist_spec = PATROL_WEIGHTING_SPECS["dist_meters"]
    assert (dist_spec.density_sum_column, dist_spec.original_unit, dist_spec.display_unit) == (
        "dist_meters",
        Unit.METER,
        Unit.KILOMETER,
    )
    assert (dist_spec.option_label, dist_spec.mode) == ("Distance", "sum")


def test_ltd_spec():
    ltd_spec = PATROL_WEIGHTING_SPECS["normalised_ltd"]
    assert ltd_spec.density_sum_column == "timespan_seconds"
    assert (ltd_spec.original_unit, ltd_spec.display_unit) == (Unit.SECOND, Unit.PERCENT)
    assert (ltd_spec.option_label, ltd_spec.mode) == ("Normalised (LTD)", "ltd")


def test_get_weighting_column():
    assert get_weighting_column(weighting_spec=PATROL_WEIGHTING_SPECS["timespan_seconds"]) == "timespan_seconds"
    assert get_weighting_column(weighting_spec=PATROL_WEIGHTING_SPECS["dist_meters"]) == "dist_meters"
    assert get_weighting_column(weighting_spec=PATROL_WEIGHTING_SPECS["normalised_ltd"]) == "timespan_seconds"


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


def test_labeled_weighting_replaces_enum_with_labeled_options():
    schema = {"enum": ["timespan_seconds", "dist_meters", "normalised_ltd"]}
    labeled_weighting(PATROL_WEIGHTING_SPECS)(schema)
    assert "enum" not in schema
    assert schema["oneOf"] == [
        {"const": "timespan_seconds", "title": "Time"},
        {"const": "dist_meters", "title": "Distance"},
        {"const": "normalised_ltd", "title": "Normalised (LTD)"},
    ]


@pytest.mark.parametrize("weighting", ["timespan_seconds", "dist_meters"])
def test_classified_track_density_sum_matches_legacy_pipeline(trajectory_gdf, meshgrid, weighting):
    spec = PATROL_WEIGHTING_SPECS[weighting]

    result = calculate_classified_track_density(
        geodataframe=trajectory_gdf,
        meshgrid=meshgrid,
        weighting_spec=spec,
    )

    legacy = calculate_feature_density(
        geodataframe=trajectory_gdf,
        meshgrid=meshgrid.copy(),
        geometry_type="line",
        sum_column=spec.density_sum_column,
    )
    legacy = normalize_density_units(df=legacy, weighting_spec=spec)
    legacy = sort_values(df=legacy, column_name="density", ascending=True, na_position="last")
    legacy = drop_nan_values_by_column(df=legacy, column_name="density")
    legacy = apply_classification(
        df=legacy,
        input_column_name="density",
        output_column_name="density_bins",
        label_options=DefaultLabels(label_ranges=True, label_decimals=1),
        classification_options=SharedArgs(scheme="equal_interval", k=10),
    )

    assert len(result) > 0
    assert list(result["density"]) == list(legacy["density"])
    assert list(result["density_bins"]) == list(legacy["density_bins"])
    assert list(result.geometry) == list(legacy.geometry)


def test_classified_track_density_ltd_matches_linear_time_density(trajectory_gdf, meshgrid):
    result = calculate_classified_track_density(
        geodataframe=trajectory_gdf,
        meshgrid=meshgrid,
        weighting_spec=PATROL_WEIGHTING_SPECS["normalised_ltd"],
    )

    expected = calculate_linear_time_density(
        trajectory_gdf=trajectory_gdf,
        meshgrid=meshgrid.copy(),
    )

    assert len(result) == len(expected)
    assert list(result["density"]) == sorted(expected["percentile"])
    assert list(result["density_bins"]) == [f"{p} %" for p in result["density"]]
    expected_geoms = expected.sort_values("percentile").geometry
    assert all(a.equals(b) for a, b in zip(result.geometry, expected_geoms))


def test_classified_track_density_ltd_custom_percentiles(trajectory_gdf, meshgrid):
    spec = set_patrol_weighting_spec(
        weighting=PatrolWeightingSelection(density_sum_column="normalised_ltd", percentiles=[50.0, 90.0, 100.0])
    )
    result = calculate_classified_track_density(
        geodataframe=trajectory_gdf,
        meshgrid=meshgrid,
        weighting_spec=spec,
    )

    expected = calculate_linear_time_density(
        trajectory_gdf=trajectory_gdf,
        meshgrid=meshgrid.copy(),
        percentiles=[50.0, 90.0, 100.0],
    )

    assert set(result["density"]) <= {50.0, 90.0, 100.0}
    assert len(result) == len(expected)
    assert list(result["density"]) == sorted(expected["percentile"])


@pytest.mark.parametrize("weighting", ["timespan_seconds", "dist_meters", "normalised_ltd"])
def test_classified_track_density_empty_trajectory(trajectory_gdf, meshgrid, weighting):
    result = calculate_classified_track_density(
        geodataframe=trajectory_gdf.iloc[0:0],
        meshgrid=meshgrid,
        weighting_spec=PATROL_WEIGHTING_SPECS[weighting],
    )
    assert result.empty
    assert "density" in result.columns
    assert "density_bins" in result.columns
