import geopandas as gpd  # type: ignore[import-untyped]
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import LineString, Point, box

from ecoscope.platform.tasks.analysis import (
    calculate_encounter_rate_grid,
    drop_inactive_grid_cells,
    mask_low_effort_cells,
)


@pytest.fixture
def meshgrid():
    """Four 1 km cells in a row along x."""
    return gpd.GeoDataFrame(geometry=[box(x, 0, x + 1000, 1000) for x in range(0, 4000, 1000)], crs="EPSG:3857")


@pytest.fixture
def events():
    """Two events in cell 0 and one in cell 2; served in 4326 to exercise reprojection."""
    return gpd.GeoDataFrame(
        {"number_of_animals": [3, None, "5"]},
        geometry=[Point(200, 500), Point(700, 500), Point(2500, 500)],
        crs="EPSG:3857",
    ).to_crs("EPSG:4326")


@pytest.fixture
def trajectories():
    """1 km of effort in cell 0, 0.5 km in cell 1, none in cells 2 and 3."""
    return gpd.GeoDataFrame(geometry=[LineString([(0, 500), (1500, 500)])], crs="EPSG:3857")


def test_encounter_rate_counts_events_per_km(meshgrid, events, trajectories):
    grid = calculate_encounter_rate_grid([events, trajectories], meshgrid)
    assert grid.crs == meshgrid.crs
    assert grid["event_count"].tolist() == [2, 0, 1, 0]
    assert grid["patrol_effort_km"].tolist() == [1.0, 0.5, 0.0, 0.0]
    # Effort but no events rates 0; events without effort, and no effort at all, are NaN.
    np.testing.assert_allclose(grid["density"], [2.0, 0.0, np.nan, np.nan], rtol=1e-6)
    assert "event_sum" not in grid


def test_encounter_rate_sums_a_column(meshgrid, events, trajectories):
    grid = calculate_encounter_rate_grid([events, trajectories], meshgrid, sum_column="number_of_animals")
    # Missing values are ignored and numeric strings coerced.
    assert grid["event_sum"].tolist() == [3.0, 0.0, 5.0, 0.0]
    assert grid["event_count"].tolist() == [2, 0, 1, 0]
    np.testing.assert_allclose(grid["density"], [3.0, 0.0, np.nan, np.nan], rtol=1e-6)


def test_encounter_rate_rejects_an_unknown_sum_column(meshgrid, events, trajectories):
    with pytest.raises(ValueError, match="'nope' not found in the events data"):
        calculate_encounter_rate_grid([events, trajectories], meshgrid, sum_column="nope")


def test_encounter_rate_with_a_skipped_layer_is_all_nan(meshgrid, events):
    grid = calculate_encounter_rate_grid([events], meshgrid)
    assert len(grid) == len(meshgrid)
    assert grid["density"].isna().all()
    assert "event_count" not in grid


def test_encounter_rate_does_not_mutate_the_meshgrid(meshgrid, events, trajectories):
    before = meshgrid.copy()
    calculate_encounter_rate_grid([events, trajectories], meshgrid)
    pd.testing.assert_frame_equal(meshgrid, before)


def test_drop_inactive_grid_cells(meshgrid, events, trajectories):
    grid = calculate_encounter_rate_grid([events, trajectories], meshgrid)
    active = drop_inactive_grid_cells(grid)
    # Cell 3 has neither events nor effort; cell 2 has an event without effort and stays.
    assert active.index.tolist() == [0, 1, 2]


def test_drop_inactive_grid_cells_custom_columns():
    df = gpd.GeoDataFrame({"n": [0, 1], "km": [0.0, 0.0]}, geometry=[Point(0, 0), Point(1, 1)])
    assert drop_inactive_grid_cells(df, event_column="n", patrol_effort_column="km").index.tolist() == [1]


def test_drop_inactive_grid_cells_without_counts_drops_everything(meshgrid, events):
    """The shape `calculate_encounter_rate_grid` returns when a layer was skipped."""
    grid = calculate_encounter_rate_grid([events], meshgrid)
    out = drop_inactive_grid_cells(grid)
    assert out.empty
    assert list(out.columns) == list(grid.columns)


@pytest.fixture
def classified_grid():
    return gpd.GeoDataFrame(
        {
            "patrol_effort_km": [1.0, 0.1, 0.5, 0.0],
            "rate_colormap": [(255, 0, 0, 255), (0, 255, 0, 255), (0, 0, 255, 255), (9, 9, 9, 255)],
            "rate_bins": ["high", "low", "mid", "none"],
        },
        geometry=[Point(i, i) for i in range(4)],
    )


def test_mask_low_effort_cells_greys_and_moves_masked_rows_last(classified_grid):
    out = mask_low_effort_cells(classified_grid)
    assert out.index.tolist() == [0, 2, 1, 3]
    assert out["rate_bins"].tolist() == ["high", "mid", "< 200 m patrol effort", "< 200 m patrol effort"]
    assert out.loc[1, "rate_colormap"] == (128, 128, 128, 255)
    assert out.loc[3, "rate_colormap"] == (128, 128, 128, 255)
    assert out.loc[0, "rate_colormap"] == (255, 0, 0, 255)
    # The input is left alone.
    assert classified_grid["rate_bins"].tolist() == ["high", "low", "mid", "none"]


def test_mask_low_effort_cells_with_a_threshold(classified_grid):
    out = mask_low_effort_cells(classified_grid, threshold_km=0.6, grey_color=(1, 2, 3, 4), grey_label="thin")
    assert out.index.tolist() == [0, 1, 2, 3]
    assert out["rate_bins"].tolist() == ["high", "thin", "thin", "thin"]
    assert out["rate_colormap"].tolist()[1:] == [(1, 2, 3, 4)] * 3


def test_mask_low_effort_cells_with_nothing_below_threshold(classified_grid):
    out = mask_low_effort_cells(classified_grid, threshold_km=0)
    pd.testing.assert_frame_equal(out, classified_grid)
