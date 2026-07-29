from collections.abc import Callable
from dataclasses import dataclass
from typing import Annotated, Literal, TypeAlias

from pydantic import Field
from wt_registry import register

from ecoscope.platform.annotations import AnyGeoDataFrame
from ecoscope.platform.tasks.transformation._unit import Unit, with_unit


@dataclass(frozen=True)
class SumWeightingSpec:
    # Per-cell column sum, equal-interval value bins.
    density_sum_column: str  # gdf column the density is summed from
    original_unit: Unit  # unit of the raw summed column
    display_unit: Unit  # unit shown on the map and in the legend title
    option_label: str  # form dropdown label and default legend title prefix
    legend_label: str | None = None  # legend title prefix when it differs from option_label
    # Bins are emitted ascending, so low sums get the first colors: green -> red.
    colormap: str = "RdYlGn_r"
    mode: Literal["sum"] = "sum"  # union discriminator


@dataclass(frozen=True)
class UDWeightingSpec:
    # Utilisation distribution (currently LTD), percentile bins.
    option_label: str  # form dropdown label and default legend title prefix
    legend_label: str | None = None  # legend title prefix when it differs from option_label
    percentiles: tuple[float, ...] | None = None  # percentile bins; None -> LTD defaults
    display_unit: Unit = Unit.PERCENT  # unit shown on the map and in the legend title
    # Lowest isopleth is the densest core, so red comes first (patrols parity).
    colormap: str = "RdYlGn"
    mode: Literal["ud"] = "ud"  # union discriminator


WeightingSpec: TypeAlias = SumWeightingSpec | UDWeightingSpec


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
        Field(
            description="Feature density output with a raw 'density' column.",
            exclude=True,
        ),
    ],
    weighting_spec: Annotated[
        SumWeightingSpec,
        Field(
            description="The weighting the density was summed from; determines the display unit.",
            exclude=True,
        ),
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
        Field(
            description="The weighting the density was summed from; determines the label and display unit.",
            exclude=True,
        ),
    ],
) -> str:
    """
    Legend title for the density map: the weighting's label and display unit.
    """
    label = weighting_spec.legend_label or weighting_spec.option_label
    return f"{label} ({weighting_spec.display_unit.value})"


@register()
def get_density_colormap(
    weighting_spec: Annotated[
        WeightingSpec,
        Field(
            description="The weighting the density was summed from; determines the colormap direction.",
            exclude=True,
        ),
    ],
) -> str:
    """
    Colormap for the density map: high sums are red, but for percentile (UD)
    weightings the lowest isopleth is the densest core, so the direction flips.
    """
    return weighting_spec.colormap


@register()
def get_weighting_column(
    weighting_spec: Annotated[
        SumWeightingSpec,
        Field(description="The weighting to read the column name from.", exclude=True),
    ],
) -> str:
    """
    The gdf column the density is summed from.
    """
    return weighting_spec.density_sum_column
