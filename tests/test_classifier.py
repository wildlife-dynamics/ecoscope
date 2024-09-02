import pytest
import pandas as pd
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
            [1.4188611699158102, 4.58113883008419, 4.58113883008419, 4.58113883008419, 6.16227766016838],
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


def test_classify_with_invalid_labels(sample_df):
    with pytest.raises(AssertionError):
        apply_classification(sample_df, input_column_name="value", labels=[0], scheme="std_mean")


def test_classify_with_invalid_scheme(sample_df):
    with pytest.raises(ValueError):
        apply_classification(sample_df, input_column_name="value", scheme="InvalidScheme")


def test_color_lookup(sample_df):

    classified = apply_classification(sample_df, input_column_name="value", scheme="equal_interval")
    cmap = "viridis"

    color_lookup = apply_color_map(classified, "value_classified", cmap)
    assert len(color_lookup) == len(classified["value_classified"])
    assert len(set(color_lookup)) == len(classified["value_classified"].unique())


def test_color_lookup_k2(sample_df):

    classified = apply_classification(sample_df, input_column_name="value", scheme="equal_interval", k=2)
    cmap = "viridis"

    color_lookup = apply_color_map(classified, "value_classified", cmap)
    assert len(color_lookup) == len(classified["value_classified"])
    assert len(set(color_lookup)) == len(classified["value_classified"].unique())


def test_color_lookup_cmap_list(movebank_relocations):
    trajectory = Trajectory.from_relocations(movebank_relocations)
    classified = apply_classification(
        trajectory, "speed_kmhr", output_column_name="speed_bins", k=6, scheme="equal_interval"
    )

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

    color_lookup = apply_color_map(classified, "speed_bins", cmap)
    assert len(color_lookup) == len(classified["speed_bins"])
    assert len(set(color_lookup)) == 6


def test_color_lookup_cmap_bad_list(movebank_relocations):
    trajectory = Trajectory.from_relocations(movebank_relocations)
    classified = apply_classification(
        trajectory, "speed_kmhr", output_column_name="speed_bins", k=6, scheme="equal_interval"
    )

    cmap = ["#1a9850"]

    with pytest.raises(AssertionError):
        apply_color_map(classified, "speed_bins", cmap)
