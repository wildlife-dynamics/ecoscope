from collections.abc import Callable
from dataclasses import dataclass
from typing import Annotated, Literal, cast

from pydantic import Field
from wt_registry import register

from ecoscope.platform.annotations import AnyDataFrame, AnyGeoDataFrame
from ecoscope.platform.tasks.transformation._unit import Unit, with_unit


@dataclass(frozen=True)
class WeightingSpec:
    density_sum_column: str  # gdf column the density is summed from
    original_unit: Unit  # unit of the raw summed column
    display_unit: Unit  # unit shown on the map and in the legend title
    option_label: str  # form dropdown label and default legend title prefix
    # "sum": per-cell column sum, equal-interval value bins;
    # "ltd": linear time density, percentile bins.
    mode: Literal["sum", "ltd"] = "sum"
    legend_label: str | None = None  # legend title prefix when it differs from option_label


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
def get_weighting_column(
    weighting_spec: Annotated[
        WeightingSpec,
        Field(description="The weighting to read the column name from.", exclude=True),
    ],
) -> str:
    """
    The gdf column the density is summed from.
    """
    return weighting_spec.density_sum_column


def _empty_density_result(meshgrid: AnyGeoDataFrame) -> AnyGeoDataFrame:
    import geopandas as gpd  # type: ignore[import-untyped]
    import pandas as pd

    return gpd.GeoDataFrame(
        {"density": pd.Series(dtype="float64"), "density_bins": pd.Series(dtype="object")},
        geometry=gpd.GeoSeries(dtype="geometry"),
        crs=meshgrid.crs,
    )


def _classified_sum_density(
    geodataframe: AnyGeoDataFrame,
    meshgrid: AnyGeoDataFrame,
    weighting_spec: WeightingSpec,
) -> AnyGeoDataFrame:
    from ecoscope.platform.tasks.analysis._calculate_feature_density import calculate_feature_density
    from ecoscope.platform.tasks.transformation._classification import (
        DefaultLabels,
        SharedArgs,
        apply_classification,
    )

    # copy: calculate_feature_density writes into the grid it is given, and the
    # same meshgrid object is shared across mapvalues groups
    result = calculate_feature_density(
        geodataframe=geodataframe,
        meshgrid=meshgrid.copy(),
        geometry_type="line",
        sum_column=weighting_spec.density_sum_column,
    )
    result = normalize_density_units(df=result, weighting_spec=weighting_spec)
    result = result.sort_values(by="density", ascending=True, na_position="last")
    result = result[result["density"].notna()]
    if result.empty:
        return _empty_density_result(meshgrid)

    return cast(
        AnyGeoDataFrame,
        apply_classification(
            df=cast(AnyDataFrame, result),
            input_column_name="density",
            output_column_name="density_bins",
            label_options=DefaultLabels(label_ranges=True, label_decimals=1),
            classification_options=SharedArgs(scheme="equal_interval", k=10),
        ),
    )


def _classified_ltd_density(
    geodataframe: AnyGeoDataFrame,
    meshgrid: AnyGeoDataFrame,
) -> AnyGeoDataFrame:
    from ecoscope import Trajectory
    from ecoscope.analysis.classifier import classify_percentile
    from ecoscope.analysis.linear_time_density import calculate_ltd
    from ecoscope.platform.tasks.analysis._time_density import LTD_DEFAULT_PERCENTILES

    density_grid = calculate_ltd(traj=Trajectory(geodataframe), grid=meshgrid.copy())
    if density_grid["density"].isna().all():
        return _empty_density_result(meshgrid)

    result = classify_percentile(
        df=density_grid,
        percentile_levels=[float(p) for p in LTD_DEFAULT_PERCENTILES],  # type: ignore[arg-type,misc]
        input_column_name="density",
    )
    result = result.dissolve(  # type: ignore[operator]
        "percentile",
        aggfunc={"density": "sum"},
        as_index=False,
    )
    result = result[result["percentile"].notna()].sort_values(by="percentile", ascending=True)
    result["density"] = result["percentile"]
    # Patrols shows the suffix via the polygon layer's legend label_suffix (" %"),
    # which is static in the spec; baking it into the labels keeps the spec
    # mode-agnostic.
    result["density_bins"] = result["percentile"].astype(str) + " %"
    return result


@register()
def calculate_classified_track_density(
    geodataframe: Annotated[
        AnyGeoDataFrame,
        Field(description="The trajectory segments to grid into a density surface.", exclude=True),
    ],
    meshgrid: Annotated[
        AnyGeoDataFrame,
        Field(description="The grid cells the density is calculated over.", exclude=True),
    ],
    weighting_spec: Annotated[
        WeightingSpec,
        Field(description="The weighting that selects the density calculation mode.", exclude=True),
    ],
) -> AnyGeoDataFrame:
    """
    Calculate a classified track density grid in the mode selected by the weighting.

    Emits a uniform shape across modes: a 'density' column with the numeric
    display value (display-unit sums for "sum" weightings, percentiles for
    "ltd") and a 'density_bins' column with ready-to-render string bin labels,
    sorted ascending with NaN cells dropped.
    """
    if geodataframe.empty:
        return _empty_density_result(meshgrid)

    if weighting_spec.mode == "ltd":
        return _classified_ltd_density(geodataframe, meshgrid)
    return _classified_sum_density(geodataframe, meshgrid, weighting_spec)
