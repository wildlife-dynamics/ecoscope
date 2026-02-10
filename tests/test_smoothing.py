import numpy as np
import pytest

from ecoscope.analysis.smoothing import SmoothingConfig, apply_smoothing


def test_create_config_spline():
    config = SmoothingConfig(method="spline")
    assert config.method == "spline"
    assert config.y_min is None
    assert config.y_max is None
    assert config.resolution == 10
    assert config.degree == 3


def test_create_config_with_y_min():
    config = SmoothingConfig(method="spline", y_min=0)
    assert config.method == "spline"
    assert config.y_min == 0
    assert config.y_max is None


def test_create_config_with_y_max():
    config = SmoothingConfig(method="spline", y_max=100)
    assert config.method == "spline"
    assert config.y_min is None
    assert config.y_max == 100


def test_create_config_with_both_clamps():
    config = SmoothingConfig(method="spline", y_min=0, y_max=100)
    assert config.method == "spline"
    assert config.y_min == 0
    assert config.y_max == 100


def test_create_config_with_resolution():
    config = SmoothingConfig(method="spline", resolution=5)
    assert config.resolution == 5


def test_create_config_with_degree():
    config = SmoothingConfig(method="spline", degree=2)
    assert config.degree == 2


def test_spline_smoothing_basic():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y = np.array([1.0, 4.0, 2.0, 5.0, 3.0])
    config = SmoothingConfig(method="spline")

    x_smooth, y_smooth = apply_smoothing(x, y, config)

    # Should have more points than original (10x interpolation)
    assert len(x_smooth) == len(x) * 10
    assert len(y_smooth) == len(x) * 10
    # x_smooth should span the same range
    assert x_smooth[0] == pytest.approx(x.min())
    assert x_smooth[-1] == pytest.approx(x.max())


def test_spline_smoothing_with_y_min():
    # Create data where spline might dip below zero
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y = np.array([10.0, 0.5, 10.0, 0.5, 10.0])
    config = SmoothingConfig(method="spline", y_min=0)

    x_smooth, y_smooth = apply_smoothing(x, y, config)

    # All smoothed values should be >= 0
    assert np.all(y_smooth >= 0)


def test_spline_smoothing_with_y_max():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y = np.array([10.0, 50.0, 10.0, 50.0, 10.0])
    config = SmoothingConfig(method="spline", y_max=60)

    x_smooth, y_smooth = apply_smoothing(x, y, config)

    # All smoothed values should be <= 60
    assert np.all(y_smooth <= 60)


def test_spline_smoothing_with_both_clamps():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y = np.array([10.0, 0.5, 50.0, 0.5, 10.0])
    config = SmoothingConfig(method="spline", y_min=0, y_max=60)

    x_smooth, y_smooth = apply_smoothing(x, y, config)

    # All smoothed values should be within bounds
    assert np.all(y_smooth >= 0)
    assert np.all(y_smooth <= 60)


def test_spline_smoothing_with_datetime():
    x = np.array(
        ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"],
        dtype="datetime64[D]",
    )
    y = np.array([1.0, 4.0, 2.0, 5.0, 3.0])
    config = SmoothingConfig(method="spline")

    x_smooth, y_smooth = apply_smoothing(x, y, config)

    # Should preserve datetime type
    assert np.issubdtype(x_smooth.dtype, "datetime64")
    assert len(x_smooth) == len(x) * 10
    assert len(y_smooth) == len(x) * 10


def test_spline_smoothing_with_python_date():
    import datetime

    x = np.array(
        [
            datetime.date(2024, 1, 1),
            datetime.date(2024, 1, 2),
            datetime.date(2024, 1, 3),
            datetime.date(2024, 1, 4),
            datetime.date(2024, 1, 5),
        ]
    )
    y = np.array([1.0, 4.0, 2.0, 5.0, 3.0])
    config = SmoothingConfig(method="spline")

    x_smooth, y_smooth = apply_smoothing(x, y, config)

    # Should convert to datetime64
    assert np.issubdtype(x_smooth.dtype, "datetime64")
    assert len(x_smooth) == len(x) * 10
    assert len(y_smooth) == len(x) * 10


def test_spline_smoothing_unsorted_input():
    # Input data is not sorted by x
    x = np.array([3.0, 1.0, 5.0, 2.0, 4.0])
    y = np.array([2.0, 1.0, 3.0, 4.0, 5.0])
    config = SmoothingConfig(method="spline")

    x_smooth, y_smooth = apply_smoothing(x, y, config)

    # Should still produce valid output (sorted internally)
    assert len(x_smooth) == len(x) * 10
    assert x_smooth[0] == pytest.approx(x.min())
    assert x_smooth[-1] == pytest.approx(x.max())
    # x_smooth should be sorted
    assert np.all(np.diff(x_smooth) >= 0)


def test_spline_smoothing_with_custom_resolution():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y = np.array([1.0, 4.0, 2.0, 5.0, 3.0])
    config = SmoothingConfig(method="spline", resolution=5)

    x_smooth, y_smooth = apply_smoothing(x, y, config)

    # Should have 5x points
    assert len(x_smooth) == len(x) * 5
    assert len(y_smooth) == len(x) * 5


def test_spline_smoothing_with_custom_degree():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y = np.array([1.0, 4.0, 2.0, 5.0, 3.0])

    # Test with linear interpolation (degree=1)
    config_linear = SmoothingConfig(method="spline", degree=1)
    x_linear, y_linear = apply_smoothing(x, y, config_linear)

    # Test with cubic spline (degree=3)
    config_cubic = SmoothingConfig(method="spline", degree=3)
    x_cubic, y_cubic = apply_smoothing(x, y, config_cubic)

    # Both should produce valid output
    assert len(x_linear) == len(x) * 10
    assert len(x_cubic) == len(x) * 10

    # Linear and cubic should produce different results
    assert not np.allclose(y_linear, y_cubic)


def test_unsupported_smoothing_method():
    x = np.array([1.0, 2.0, 3.0])
    y = np.array([1.0, 2.0, 3.0])
    # Create config with invalid method by bypassing type checking
    config = SmoothingConfig.__new__(SmoothingConfig)
    config.method = "invalid_method"
    config.y_min = None
    config.y_max = None
    config.resolution = 10
    config.degree = 3

    with pytest.raises(NotImplementedError, match="Unsupported smoothing method"):
        apply_smoothing(x, y, config)


def test_spline_smoothing_insufficient_points():
    # Cubic spline (degree=3) requires at least 4 points
    x = np.array([3.0, 1.0, 2.0])  # Only 3 points, unsorted
    y = np.array([30.0, 10.0, 20.0])
    config = SmoothingConfig(method="spline", degree=3)

    x_smooth, y_smooth = apply_smoothing(x, y, config)

    # Should return original data sorted, unchanged
    assert len(x_smooth) == 3
    assert len(y_smooth) == 3
    np.testing.assert_array_equal(x_smooth, [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(y_smooth, [10.0, 20.0, 30.0])


def test_spline_smoothing_exact_minimum_points():
    # Cubic spline (degree=3) requires exactly 4 points minimum
    x = np.array([1.0, 2.0, 3.0, 4.0])
    y = np.array([1.0, 4.0, 2.0, 5.0])
    config = SmoothingConfig(method="spline", degree=3)

    x_smooth, y_smooth = apply_smoothing(x, y, config)

    # Should successfully smooth with exactly k+1 points
    assert len(x_smooth) == len(x) * 10
    assert len(y_smooth) == len(x) * 10
