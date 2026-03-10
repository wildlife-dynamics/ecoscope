from dataclasses import dataclass
from enum import Enum
from typing import Annotated

from pydantic import Field
from pydantic.json_schema import SkipJsonSchema
from wt_registry import register


class Unit(Enum):
    METER = "m"
    KILOMETER = "km"
    SECOND = "s"
    HOUR = "h"
    DAY = "d"
    METERS_PER_SECOND = "m/s"
    KILOMETERS_PER_HOUR = "km/h"

    def __str__(self) -> str:
        return self.value


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
