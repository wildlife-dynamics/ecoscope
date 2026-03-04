import astropy.units as u  # type: ignore[import-untyped]
import pytest
from ecoscope.platform.tasks.transformation._unit import Unit, with_unit


def test_apply_unit_conversion():
    data = with_unit(1.0, Unit.METER, Unit.KILOMETER)
    assert data.value == 0.001
    assert data.unit == Unit.KILOMETER


def test_apply_speed_conversion():
    data = with_unit(1, Unit.METERS_PER_SECOND, Unit.KILOMETERS_PER_HOUR)
    assert data.value == pytest.approx(3.6)
    assert data.unit == Unit.KILOMETERS_PER_HOUR


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
