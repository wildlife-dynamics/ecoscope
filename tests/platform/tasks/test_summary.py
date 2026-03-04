from importlib.resources import files

import pandas as pd
import pytest
from ecoscope.platform.mock_loaders import load_parquet
from ecoscope.platform.tasks.analysis._summary import (
    SummaryParam,
    summarize_df,
)


@pytest.fixture
def sample_dataframe():
    data = {"A": [1, 2, 3, 4, 5], "B": [5, 4, 3, 2, 1], "C": [10, 20, 30, 40, 50]}
    return pd.DataFrame(data)


@pytest.fixture
def trajectories():
    return load_parquet(
        files("ecoscope.platform.tasks.preprocessing")
        / "relocations-to-trajectory.example-return.parquet"
    )


def test_summarize_df_sum(sample_dataframe):
    summary_params = [
        SummaryParam(display_name="Sum of A", aggregator="sum", column="A"),
        SummaryParam(display_name="Min of A", aggregator="min", column="A"),
        SummaryParam(display_name="Max of A", aggregator="max", column="A"),
        SummaryParam(display_name="Mean of A", aggregator="mean", column="A"),
        SummaryParam(display_name="Median of A", aggregator="median", column="A"),
        SummaryParam(display_name="Count of A", aggregator="count", column="A"),
        SummaryParam(display_name="Sum of B", aggregator="sum", column="B"),
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
    summary_params = [
        SummaryParam(display_name="Sum of A", aggregator="sum", column="A")
    ]
    result = summarize_df(sample_dataframe, summary_params, groupby_cols=["Group"])
    assert result.loc["X", "Sum of A"] == 3
    assert result.loc["Y", "Sum of A"] == 12


def test_summarize_df_with_units(sample_dataframe):
    summary_params = [
        SummaryParam(
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


def test_summarize_df_with_missing_column(sample_dataframe):
    with pytest.raises(ValueError):
        SummaryParam(display_name="Sum of A", aggregator="sum")


def test_summarize_df_with_missing_unit(sample_dataframe):
    with pytest.raises(ValueError):
        SummaryParam(
            display_name="Sum of A", aggregator="sum", column="A", original_unit="m"
        )


def test_summarize_df_night_day_ratio(trajectories):
    summary_params = [
        SummaryParam(
            display_name="Night Day Ratio",
            aggregator="night_day_ratio",
            decimal_places=2,
        ),
        SummaryParam(
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
    assert result.loc[0, "Night Day Ratio"] == 1.03
