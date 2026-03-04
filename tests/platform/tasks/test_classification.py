from importlib.resources import files

import geopandas as gpd  # type: ignore[import-untyped]
import numpy as np
import pandas as pd
import pytest
from ecoscope.platform.tasks.transformation._classification import (
    CustomLabels,
    DefaultLabels,
    MaxBreaksArgs,
    NaturalBreaksArgs,
    SharedArgs,
    StdMeanArgs,
    apply_classification,
    apply_color_map,
    classify_seasons,
)


@pytest.fixture
def test_df():
    return pd.DataFrame({"column_name": [5, 3, 1, 6, 5, 9]})


def test_color_map():
    df = pd.DataFrame({"column_name": ["A", "B", "A", "C", "B", "C"]})
    result = apply_color_map(df, "column_name", ["#FF0000", "#00FF00", "#0000FF"])

    assert "column_name_colormap" in result.columns

    color_mapping = {
        "A": (255, 0, 0, 255),
        "B": (0, 255, 0, 255),
        "C": (0, 0, 255, 255),
    }
    for _, row in result.iterrows():
        np.testing.assert_array_equal(
            row["column_name_colormap"],
            color_mapping[row["column_name"]],
        )


@pytest.mark.parametrize(
    "classification_args, label_args",
    [
        (
            SharedArgs(k=3),
            DefaultLabels(label_ranges=True, label_decimals=0),
        ),
        (
            MaxBreaksArgs(k=4, min_diff=20),
            DefaultLabels(label_prefix="_", label_suffix="_"),
        ),
        (
            NaturalBreaksArgs(k=4, initial=3),
            CustomLabels(labels=["One", "Two", "Three", "Four"]),
        ),
        (StdMeanArgs(multiples=[-1, 1], anchor=False), DefaultLabels()),
    ],
)
def test_apply_classification(test_df, classification_args, label_args):
    result = apply_classification(
        df=test_df,
        input_column_name="column_name",
        output_column_name="classified",
        label_options=label_args,
        classification_options=classification_args,
    )

    assert "classified" in result.columns


def test_classify_seasons():
    example_traj_df_path = (
        files("ecoscope.platform.tasks.preprocessing")
        / "relocations-to-trajectory.example-return.parquet"
    )
    trajectory = gpd.read_parquet(example_traj_df_path)

    example_season_df_path = (
        files("ecoscope.platform.tasks.io")
        / "determine-season-windows.example-return.parquet"
    )
    season_windows = pd.read_parquet(example_season_df_path)

    result = classify_seasons(trajectory, season_windows)

    assert "season" in result.columns
