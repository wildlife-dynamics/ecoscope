import pytest
import pandas as pd
import numpy as np
from ecoscope.base import Trajectory
from ecoscope.analysis.classifier import apply_classification, apply_color_map


@pytest.fixture
def sample_df():
    return pd.DataFrame(
        data={"value": [1, 2, 3, 4, 5]},
        index=["A", "B", "C", "D", "E"],
    )


@pytest.mark.parametrize(
    "scheme,kwargs,expected",
    [
        ("equal_interval", {"k": 2}, [3, 3, 3, 5, 5]),
        ("quantile", {"k": 2}, [3, 3, 3, 5, 5]),
        (
            "std_mean",
            {"multiples": [-2, -1, 1, 2]},
            [1.4, 4.6, 4.6, 4.6, 6.2],
        ),
        ("max_breaks", {"k": 4}, [2.5, 2.5, 3.5, 4.5, 5.0]),
        ("fisher_jenks", {"k": 5}, [1.0, 2.0, 3.0, 4.0, 5.0]),
    ],
)
def test_classify(sample_df, scheme, kwargs, expected):
    result = apply_classification(sample_df, "value", scheme=scheme, **kwargs)
    pd.testing.assert_index_equal(result.index, sample_df.index)
    assert result["value_classified"].values.tolist() == expected, f"Failed on scheme {scheme}"


def test_classify_with_labels(sample_df):
    result = apply_classification(sample_df, input_column_name="value", labels=["1", "2"], scheme="equal_interval", k=2)
    assert result["value_classified"].values.tolist() == ["1", "1", "1", "2", "2"]


def test_classify_with_labels_prefix_suffix(sample_df):
    result = apply_classification(
        sample_df,
        input_column_name="value",
        labels=["1", "2"],
        label_prefix="_",
        label_suffix="_",
        scheme="equal_interval",
        k=2,
    )
    assert result["value_classified"].values.tolist() == ["_1_", "_1_", "_1_", "_2_", "_2_"]


def test_classify_with_invalid_labels(sample_df):
    with pytest.raises(AssertionError):
        apply_classification(sample_df, input_column_name="value", labels=[0], scheme="std_mean")


def test_classify_with_invalid_scheme(sample_df):
    with pytest.raises(ValueError):
        apply_classification(sample_df, input_column_name="value", scheme="InvalidScheme")


@pytest.mark.parametrize("cmap", ["viridis", "RdYlGn"])
def test_apply_colormap(sample_df, cmap):
    apply_classification(sample_df, input_column_name="value", scheme="equal_interval")
    apply_color_map(sample_df, "value_classified", cmap, output_column_name="colormap")

    assert len(sample_df["colormap"].unique()) == len(sample_df["value_classified"].unique())


def test_apply_colormap_with_nan():
    df = pd.DataFrame(
        data={"value": [1, 2, 3, 4, np.nan]},
        index=["A", "B", "C", "D", "E"],
    )
    apply_color_map(df, "value", "viridis", output_column_name="colormap")

    assert len(df["colormap"].unique()) == len(df["value"].unique())
    assert df.loc["E"]["colormap"] == (0, 0, 0, 0)


def test_apply_colormap_k2(sample_df):
    apply_classification(sample_df, input_column_name="value", scheme="equal_interval", k=2)
    cmap = "viridis"
    apply_color_map(sample_df, "value_classified", cmap, output_column_name="colormap")

    assert len(sample_df["colormap"].unique()) == len(sample_df["value_classified"].unique())


def test_apply_colormap_user_defined(movebank_relocations):
    trajectory = Trajectory.from_relocations(movebank_relocations)
    apply_classification(trajectory, "speed_kmhr", output_column_name="speed_bins", k=6, scheme="equal_interval")

    # With len(cmap)==7 we're also testing that the input cmap can be larger than the number of categories
    cmap = [
        "#1a9850",
        "#91cf60",
        "#d9ef8b",
        "#fee08b",
        "#fc8d59",
        "#d73027",
        "#FFFFFF",
    ]

    apply_color_map(trajectory, "speed_bins", cmap)
    assert len(trajectory["speed_bins_colormap"].unique()) == len(trajectory["speed_bins"].unique())


def test_apply_colormap_cmap_user_defined_bad(movebank_relocations):
    trajectory = Trajectory.from_relocations(movebank_relocations)
    classified = apply_classification(
        trajectory, "speed_kmhr", output_column_name="speed_bins", k=6, scheme="equal_interval"
    )

    cmap = ["#1a9850"]

    with pytest.raises(AssertionError):
        apply_color_map(classified, "speed_bins", cmap)


def test_classify_with_ranges(sample_df):
    result = apply_classification(sample_df, input_column_name="value", scheme="equal_interval", label_ranges=True, k=5)
    assert result["value_classified"].values.tolist() == [
        "1.0 - 1.8",
        "1.8 - 2.6",
        "2.6 - 3.4",
        "3.4 - 4.2",
        "4.2 - 5.0",
    ]
