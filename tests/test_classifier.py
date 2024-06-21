import pytest

from ecoscope.analysis.classifier import apply_classification


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
def test_classify_data(scheme, kwargs, expected):
    y = [1, 2, 3, 4, 5]
    result = apply_classification(y, scheme=scheme, **kwargs)
    assert result == expected, f"Failed on scheme {scheme}"


def test_classify_with_labels():
    y = [1, 2, 3, 4, 5]
    result = apply_classification(y, labels=["1", "2"], scheme="equal_interval", k=2)
    assert result == ["1", "1", "1", "2", "2"]


def test_classify_with_invalid_labels():
    y = [1, 2, 3, 4, 5]
    with pytest.raises(AssertionError):
        apply_classification(y, labels=[0], scheme="std_mean")


def test_classify_with_invalid_scheme():
    y = [1, 2, 3, 4, 5]
    with pytest.raises(ValueError):
        apply_classification(y, scheme="InvalidScheme")
