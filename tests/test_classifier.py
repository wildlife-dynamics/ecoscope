import pytest
import pandas as pd
import numpy as np
from ecoscope import Trajectory
from ecoscope.analysis.classifier import apply_classification, apply_color_map, classify_percentile


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


def test_apply_mpl_colormap_loops():
    more_classes_than_colors = pd.DataFrame(
        data={"value": ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]},
        index=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    )
    apply_color_map(more_classes_than_colors, "value", "Accent", output_column_name="colormap")

    assert len(more_classes_than_colors["colormap"].unique()) != len(more_classes_than_colors["value"].unique())


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
    apply_classification(trajectory.gdf, "speed_kmhr", output_column_name="speed_bins", k=6, scheme="equal_interval")

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

    apply_color_map(trajectory.gdf, "speed_bins", cmap)
    assert len(trajectory.gdf["speed_bins_colormap"].unique()) == len(trajectory.gdf["speed_bins"].unique())


def test_apply_colormap_user_defined_loops(movebank_relocations):
    trajectory = Trajectory.from_relocations(movebank_relocations)
    apply_classification(trajectory.gdf, "speed_kmhr", output_column_name="speed_bins", k=6, scheme="equal_interval")

    # With len(cmap)==7 we're also testing that the input cmap can be larger than the number of categories
    cmap = [
        "#1a9850",
        "#91cf60",
        "#d9ef8b",
        "#fee08b",
    ]

    apply_color_map(trajectory.gdf, "speed_bins", cmap)
    assert len(trajectory.gdf["speed_bins_colormap"].unique()) == len(cmap)


def test_classify_with_ranges(sample_df):
    result = apply_classification(sample_df, input_column_name="value", scheme="equal_interval", label_ranges=True, k=5)
    assert result["value_classified"].values.tolist() == [
        "1.0 - 1.8",
        "1.8 - 2.6",
        "2.6 - 3.4",
        "3.4 - 4.2",
        "4.2 - 5.0",
    ]


def test_apply_colormap_numeric_with_single_value():
    df = pd.DataFrame(
        data={"value": [1, np.nan, np.nan, np.nan, np.nan]},
        index=["A", "B", "C", "D", "E"],
    )
    apply_color_map(df, "value", "viridis", output_column_name="colormap")

    assert len(df["colormap"].unique()) == len(df["value"].unique())
    assert df.loc["A"]["colormap"] != (0, 0, 0, 0)
    assert df.loc["E"]["colormap"] == (0, 0, 0, 0)


def test_apply_colormap_numeric_with_float_range():
    df = pd.DataFrame(
        data={"value": [0.99, 0.3, 0.001, 0.63, np.nan]},
        index=["A", "B", "C", "D", "E"],
    )
    apply_color_map(df, "value", "viridis", output_column_name="colormap")

    assert len(df["colormap"].unique()) == len(df["value"].unique())


def test_apply_colormap_numeric_nan_only():
    df = pd.DataFrame(
        data={"value": [np.nan, np.nan]},
        index=["A", "B"],
    )
    apply_color_map(df, "value", "viridis", output_column_name="colormap")

    assert len(df["colormap"].unique()) == len(df["value"].unique())
    assert df.loc["A"]["colormap"] == (0, 0, 0, 0)
    assert df.loc["B"]["colormap"] == (0, 0, 0, 0)


@pytest.mark.parametrize(
    "values",
    [
        # [np.nan, 2.0],
        # [np.nan, 1.0, 10.0, 9.0],
        # [np.nan, 2.0],
        # [np.nan, 2.0],
        # [np.nan, 2.0, 20.0],
        [np.nan],
        # [np.nan, 2.0],
        # [np.nan, 1.0, 31.0],
        # [np.nan, 27.0, 1.0],
        # [np.nan, 204.0, 438.0, 345.0, 116.0, 1.0],
        # [np.nan, 3.0, 4.0],
    ],
)
def test_apply_colormap_leaf(values):
    df = pd.DataFrame(
        data={"value": values},
    )
    apply_color_map(df, "value", "RdYlGn_r", output_column_name="colormap")

    # assert len(df["colormap"].unique()) == len(df["value"].unique())


@pytest.mark.parametrize(
    "percentile_levels,expected_values",
    [
        ([10, 50, 90], [90, 90, 50, 50, 10]),
        ([90], [90, 90, 90, 90, 90]),
        ([10], [np.nan, np.nan, np.nan, np.nan, 10]),
        ([], [90, 90, 50, 50, 10]),
    ],
)
def test_classify_percentile(sample_df, percentile_levels, expected_values):
    if percentile_levels:
        sample_df["percentile"] = expected_values

    test = classify_percentile(
        sample_df,
        percentile_levels=percentile_levels,
        input_column_name="value",
    )

    pd.testing.assert_frame_equal(test, sample_df)
