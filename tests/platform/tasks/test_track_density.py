from importlib.resources import files

import geopandas as gpd  # type: ignore[import-untyped]
import pytest

from ecoscope.platform.tasks.analysis import (
    calculate_classified_track_density,
    calculate_feature_density,
    calculate_linear_time_density,
    normalize_density_units,
    set_patrol_weighting_spec,
)
from ecoscope.platform.tasks.analysis._create_meshgrid import create_meshgrid
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


@pytest.mark.parametrize("weighting", ["timespan_seconds", "dist_meters", "normalised_ltd"])
def test_classified_track_density_disjoint_grid(trajectory_gdf, meshgrid, weighting):
    # a grid the trajectory never touches -> all-NaN densities -> empty result shape
    disjoint = meshgrid.copy()
    disjoint.geometry = disjoint.geometry.translate(xoff=1e6, yoff=1e6)
    result = calculate_classified_track_density(
        geodataframe=trajectory_gdf,
        meshgrid=disjoint,
        weighting_spec=PATROL_WEIGHTING_SPECS[weighting],
    )
    assert result.empty
    assert "density" in result.columns
    assert "density_bins" in result.columns
