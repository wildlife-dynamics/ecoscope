from collections.abc import Callable
from dataclasses import dataclass
from typing import Annotated

from pydantic import Field
from wt_registry import register

from ecoscope.platform.annotations import AnyGeoDataFrame
from ecoscope.platform.tasks.transformation._unit import Unit, with_unit


@dataclass(frozen=True)
class WeightingSpec:
    column: str  # gdf column the density is summed from
    original_unit: Unit  # unit of the raw summed column
    display_unit: Unit  # unit shown on the map and in the legend title
    option_label: str  # form dropdown label and default legend title prefix


def labeled_weighting(specs: dict[str, WeightingSpec]) -> Callable[[dict], None]:
    """Field-level json_schema_extra factory: swap the Literal's bare enum for labeled options."""

    def _apply(schema: dict) -> None:
        schema.pop("enum", None)
        schema["oneOf"] = [{"const": v, "title": spec.option_label} for v, spec in specs.items()]

    return _apply


@register()
def normalize_density_units(
    df: Annotated[
        AnyGeoDataFrame,
        Field(description="Feature density output with a raw 'density' column.", exclude=True),
    ],
    weighting_spec: Annotated[
        WeightingSpec,
        Field(description="The weighting the density was summed from; determines the display unit.", exclude=True),
    ],
) -> AnyGeoDataFrame:
    """
    Convert a raw density column from the weighting's original unit to its display unit.
    """
    df["density"] = df["density"] * with_unit(1.0, weighting_spec.original_unit, weighting_spec.display_unit).value
    return df


@register()
def get_density_legend_title(
    weighting_spec: Annotated[
        WeightingSpec,
        Field(description="The weighting the density was summed from; determines the display unit.", exclude=True),
    ],
    title_prefix: Annotated[
        str | None,
        Field(description="Legend title text preceding the display unit; defaults to the weighting's label."),
    ] = None,
) -> str:
    """
    Legend title for the density map, including the display unit.
    """
    return f"{title_prefix or weighting_spec.option_label} ({weighting_spec.display_unit.value})"


@register()
def get_weighting_column(
    weighting_spec: Annotated[
        WeightingSpec,
        Field(description="The weighting to read the column name from.", exclude=True),
    ],
) -> str:
    """
    The gdf column the density is summed from.
    """
    return weighting_spec.column
