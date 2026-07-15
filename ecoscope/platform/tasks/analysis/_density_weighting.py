from collections.abc import Callable
from dataclasses import dataclass

from ecoscope.platform.annotations import AnyGeoDataFrame
from ecoscope.platform.tasks.transformation._unit import Unit, with_unit


@dataclass(frozen=True)
class WeightingSpec:
    original_unit: Unit  # unit of the raw summed column
    display_unit: Unit  # unit shown on the map
    option_label: str  # form dropdown label
    unit_word: str  # spelled-out unit for the legend title


def labeled_weighting(specs: dict[str, WeightingSpec]) -> Callable[[dict], None]:
    """Field-level json_schema_extra factory: swap the Literal's bare enum for labeled options."""

    def _apply(schema: dict) -> None:
        schema.pop("enum", None)
        schema["oneOf"] = [{"const": v, "title": spec.option_label} for v, spec in specs.items()]

    return _apply


def normalize_density(df: AnyGeoDataFrame, spec: WeightingSpec) -> AnyGeoDataFrame:
    """Convert a raw 'density' column from the spec's original unit to its display unit."""
    df["density"] = df["density"] * with_unit(1.0, spec.original_unit, spec.display_unit).value
    return df


def density_legend_title(spec: WeightingSpec, title_prefix: str) -> str:
    """Legend title for a density map, including the display unit."""
    return f"{title_prefix} ({spec.unit_word})"
