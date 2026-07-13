from importlib.resources import files

import geopandas as gpd  # type: ignore[import-untyped]
import pandas as pd
import pytest
from shapely.geometry import LineString

from ecoscope.platform.mock_loaders import load_parquet
from ecoscope.platform.tasks.analysis._summary import (
    CoverageSummaryParam,
    NightDayRatioSummaryParam,
    NumericSummaryParam,
    TallySummaryParam,
    _coverage_area_km2,
    summarize_df,
)


@pytest.fixture
def sample_dataframe():
    data = {"A": [1, 2, 3, 4, 5], "B": [5, 4, 3, 2, 1], "C": [10, 20, 30, 40, 50]}
    return pd.DataFrame(data)


@pytest.fixture
def coverage_trajectories():
    # Two rangers, each with two overlapping LineString segments near the equator.
    return gpd.GeoDataFrame(
        {
            "ranger": ["A", "A", "B", "B"],
            "geometry": [
                LineString([(0.0, 0.0), (0.1, 0.0)]),
                LineString([(0.05, 0.0), (0.15, 0.0)]),  # overlaps the first
                LineString([(1.0, 1.0), (1.1, 1.0)]),
                LineString([(1.1, 1.0), (1.2, 1.0)]),
            ],
        },
        crs="EPSG:4326",
    )


@pytest.fixture
def trajectories():
    return load_parquet(
        files("ecoscope.platform.tasks.preprocessing") / "relocations-to-trajectory.example-return.parquet"
    )


def test_summarize_df_sum(sample_dataframe):
    summary_params = [
        NumericSummaryParam(display_name="Sum of A", aggregator="sum", column="A"),
        NumericSummaryParam(display_name="Min of A", aggregator="min", column="A"),
        NumericSummaryParam(display_name="Max of A", aggregator="max", column="A"),
        NumericSummaryParam(display_name="Mean of A", aggregator="mean", column="A"),
        NumericSummaryParam(display_name="Median of A", aggregator="median", column="A"),
        TallySummaryParam(display_name="Count of A", aggregator="count", column="A"),
        NumericSummaryParam(display_name="Sum of B", aggregator="sum", column="B"),
    ]
    result = summarize_df(sample_dataframe, summary_params)
    assert result.loc[0, "Sum of A"] == 15
    assert result.loc[0, "Min of A"] == 1
    assert result.loc[0, "Max of A"] == 5
    assert result.loc[0, "Mean of A"] == 3
    assert result.loc[0, "Median of A"] == 3
    assert result.loc[0, "Count of A"] == 5
    assert result.loc[0, "Sum of B"] == 15


def test_summarize_df_groupby(sample_dataframe):
    sample_dataframe["Group"] = ["X", "X", "Y", "Y", "Y"]
    summary_params = [NumericSummaryParam(display_name="Sum of A", aggregator="sum", column="A")]
    result = summarize_df(sample_dataframe, summary_params, groupby_cols=["Group"])
    assert result.loc["X", "Sum of A"] == 3
    assert result.loc["Y", "Sum of A"] == 12


def test_summarize_df_with_units(sample_dataframe):
    summary_params = [
        NumericSummaryParam(
            display_name="Sum of A",
            aggregator="sum",
            column="A",
            original_unit="m",
            new_unit="km",
            decimal_places=3,
        )
    ]
    result = summarize_df(sample_dataframe, summary_params)
    assert result.loc[0, "Sum of A"] == 0.015


def test_summarize_df_decimal_places_zero():
    df = pd.DataFrame({"A": [1, 2, 2]})
    summary_params = [NumericSummaryParam(display_name="Mean of A", aggregator="mean", column="A", decimal_places=0)]
    result = summarize_df(df, summary_params)
    assert result.loc[0, "Mean of A"] == 2.0


def test_summarize_df_with_missing_column(sample_dataframe):
    with pytest.raises(ValueError):
        NumericSummaryParam(display_name="Sum of A", aggregator="sum")


def test_summarize_df_with_missing_unit(sample_dataframe):
    with pytest.raises(ValueError):
        NumericSummaryParam(display_name="Sum of A", aggregator="sum", column="A", original_unit="m")


def test_summarize_df_night_day_ratio(trajectories):
    summary_params = [
        NightDayRatioSummaryParam(
            display_name="Night Day Ratio",
            aggregator="night_day_ratio",
            decimal_places=2,
        ),
        NumericSummaryParam(
            display_name="Total Dist Km",
            aggregator="sum",
            column="dist_meters",
            original_unit="m",
            new_unit="km",
            decimal_places=2,
        ),
    ]
    result = summarize_df(trajectories, summary_params)
    assert result.loc[0, "Total Dist Km"] == 2242.49
    assert result.loc[0, "Night Day Ratio"] == 1.02


def test_summarize_df_coverage_merged_le_unmerged(coverage_trajectories):
    summary_params = [
        CoverageSummaryParam(display_name="Merged", aggregator="coverage_area", merged=True, decimal_places=6),
        CoverageSummaryParam(display_name="Unmerged", aggregator="coverage_area", merged=False, decimal_places=6),
    ]
    result = summarize_df(coverage_trajectories, summary_params)
    assert result.loc[0, "Merged"] > 0
    assert result.loc[0, "Merged"] <= result.loc[0, "Unmerged"]


def test_summarize_df_coverage_scales_with_swath(coverage_trajectories):
    def unmerged(swath):
        params = [
            CoverageSummaryParam(
                display_name="Unmerged",
                aggregator="coverage_area",
                merged=False,
                swath_width_meters=swath,
                decimal_places=6,
            )
        ]
        return summarize_df(coverage_trajectories, params).loc[0, "Unmerged"]

    # Buffered-line area is dominated by length * width, so doubling the swath
    # roughly doubles the covered area.
    assert unmerged(1000.0) == pytest.approx(2 * unmerged(500.0), rel=0.05)


def test_coverage_area_km2_empty():
    empty = gpd.GeoDataFrame({"geometry": []}, crs="EPSG:4326")
    assert _coverage_area_km2(empty, 500.0, merged=True) == 0.0


def test_summarize_df_coverage_groupby(coverage_trajectories):
    summary_params = [
        CoverageSummaryParam(display_name="Merged", aggregator="coverage_area", merged=True, decimal_places=6),
    ]
    result = summarize_df(coverage_trajectories, summary_params, groupby_cols=["ranger"])
    assert len(result) == 2  # one row per ranger
    assert (result["Merged"] > 0).all()
