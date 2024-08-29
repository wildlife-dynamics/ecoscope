import pytest
import pandas as pd
from ecoscope.analysis.classifier import apply_classification, create_color_dict


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
    assert result.values.tolist() == expected, f"Failed on scheme {scheme}"


def test_classify_with_labels(sample_df):
    result = apply_classification(sample_df, column_name="value", labels=["1", "2"], scheme="equal_interval", k=2)
    assert result.to_list() == ["1", "1", "1", "2", "2"]


def test_classify_with_invalid_labels(sample_df):
    with pytest.raises(AssertionError):
        apply_classification(sample_df, column_name="value", labels=[0], scheme="std_mean")


def test_classify_with_invalid_scheme(sample_df):
    with pytest.raises(ValueError):
        apply_classification(sample_df, column_name="value", scheme="InvalidScheme")


def test_color_dict(sample_df):

    classified = apply_classification(sample_df, column_name="value", scheme="equal_interval")
    cmap = "viridis"

    color_dict = create_color_dict(classified, cmap)

    assert len(classified) == len(sample_df["value"])
    # check that our classification bins are the keys of the color_dict
    assert classified.values.tolist() == list(color_dict.keys())


def test_color_dict_k2(sample_df):

    classified = apply_classification(sample_df, column_name="value", scheme="equal_interval", k=2)
    cmap = "viridis"

    color_dict = create_color_dict(classified, cmap)

    assert len(classified) == len(sample_df["value"])
    # check that our classification bins are the keys of the color_dict
    assert classified.unique().tolist() == list(color_dict.keys())


def test_speed_parity(movebank_relocations):
    trajectory = movebank_relocations.trajectories.from_relocations()
    classified = apply_classification(trajectory, "speed_kmhr", k=6, scheme="equal_interval")

    cmap = [
        "#1a9850",
        "#91cf60",
        "#d9ef8b",
        "#fee08b",
        "#fc8d59",
        "#d73027",
    ]

    color_dict = create_color_dict(classified, cmap)
    assert len(classified) == len(trajectory["speed_kmhr"])
    assert classified.unique().tolist() == list(color_dict.keys())
