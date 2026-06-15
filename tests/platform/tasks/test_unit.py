import astropy.units as u  # type: ignore[import-untyped]
import pytest

from ecoscope.platform.tasks.transformation._unit import Unit, is_linear_unit_conversion, with_unit


def test_apply_unit_conversion():
    data = with_unit(1.0, Unit.METER, Unit.KILOMETER)
    assert data.value == 0.001
    assert data.unit == Unit.KILOMETER


def test_apply_speed_conversion():
    data = with_unit(1, Unit.METERS_PER_SECOND, Unit.KILOMETERS_PER_HOUR)
    assert data.value == pytest.approx(3.6)
    assert data.unit == Unit.KILOMETERS_PER_HOUR


def test_apply_area_conversion():
    data = with_unit(1_000_000, Unit.SQUARE_METER, Unit.SQUARE_KILOMETER)
    assert data.value == pytest.approx(1.0)
    assert data.unit == Unit.SQUARE_KILOMETER


def test_apply_unit_no_unit():
    data = with_unit(1)
    assert data.value == 1
    assert data.unit is None


def test_apply_unit_no_conversion():
    data = with_unit(1, Unit.METER)
    assert data.value == 1
    assert data.unit == Unit.METER


def test_invalid_conversion():
    with pytest.raises(u.UnitConversionError):
        with_unit(1.0, Unit.METERS_PER_SECOND, Unit.METER)


@pytest.mark.parametrize(
    "original, new, expected",
    [
        (Unit.METER, Unit.KILOMETER, True),
        (Unit.METERS_PER_SECOND, Unit.KILOMETERS_PER_HOUR, True),
        (Unit.SECOND, Unit.HOUR, True),
        # Cross-dimension is still "linear" from this helper's POV — astropy raises at conversion time.
        (Unit.METER, Unit.SECOND, True),
        # None on either side short-circuits to True (no conversion).
        (None, Unit.METER, True),
        (Unit.METER, None, True),
        (None, None, True),
    ],
)
def test_is_linear_unit_conversion_linear_cases(original, new, expected):
    assert is_linear_unit_conversion(original, new) is expected
