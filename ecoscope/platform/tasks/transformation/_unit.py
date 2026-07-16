from dataclasses import dataclass
from enum import Enum
from typing import Annotated

from pydantic import Field, JsonValue
from pydantic.json_schema import SkipJsonSchema
from wt_registry import register


class Unit(Enum):
    METER = "m"
    KILOMETER = "km"
    SQUARE_METER = "m²"
    SQUARE_KILOMETER = "km²"
    SECOND = "s"
    HOUR = "h"
    DAY = "d"
    METERS_PER_SECOND = "m/s"
    KILOMETERS_PER_HOUR = "km/h"

    def __str__(self) -> str:
        return self.value


# Human-readable labels per unit; oneOf const/title pairs render as a labeled
# dropdown.
UNIT_LABELS: dict[Unit, str] = {
    Unit.METER: "Meters (m)",
    Unit.KILOMETER: "Kilometers (km)",
    Unit.SQUARE_METER: "Square Meters (m²)",
    Unit.SQUARE_KILOMETER: "Square Kilometers (km²)",
    Unit.SECOND: "Seconds (s)",
    Unit.HOUR: "Hours (h)",
    Unit.DAY: "Days (d)",
    Unit.METERS_PER_SECOND: "Meters per Second (m/s)",
    Unit.KILOMETERS_PER_HOUR: "Kilometers per Hour (km/h)",
}

UNIT_OPTIONS: list[JsonValue] = [{"const": unit.value, "title": title} for unit, title in UNIT_LABELS.items()]


def labeled_units(*units: Unit):
    """Field-level json_schema_extra: swap a Literal's bare enum for labeled options."""

    def apply(schema: dict) -> None:
        schema.pop("enum", None)
        schema["oneOf"] = [{"const": u.value, "title": UNIT_LABELS[u]} for u in units]

    return apply


def is_linear_unit_conversion(original_unit: "Unit | None", new_unit: "Unit | None") -> bool:
    """
    True if converting between these units is a pure multiplication (so a single
    factor probed at value=1 can be broadcast across an array). False for
    log-scale units like dB/mag where astropy applies a nonlinear transform.
    """
    import astropy.units as u  # type: ignore[import-untyped]
    from astropy.units.function.mixin import FunctionMixin  # type: ignore[import-untyped]

    if original_unit is None or new_unit is None:
        return True

    a = u.Unit(original_unit.value)
    b = u.Unit(new_unit.value)
    return not (isinstance(a, FunctionMixin) or isinstance(b, FunctionMixin))


@dataclass
class Quantity:
    value: int | float
    unit: Unit | None = None


@register()
def with_unit(
    value: Annotated[float, Field(description="The original value.")],
    original_unit: Annotated[
        Unit | SkipJsonSchema[None],
        Field(description="The original unit of measurement."),
    ] = None,
    new_unit: Annotated[
        Unit | SkipJsonSchema[None],
        Field(description="The unit to convert to."),
    ] = None,
) -> Annotated[Quantity, Field(description="The value with an optional unit.")]:
    if not original_unit:
        return Quantity(value=value)

    if not new_unit:
        return Quantity(value=value, unit=original_unit)

    import astropy.units as u  # type: ignore[import-untyped]

    original = value * u.Unit(original_unit.value)
    new_quantity = original.to(u.Unit(new_unit.value))
    return Quantity(value=new_quantity.value, unit=new_unit)
