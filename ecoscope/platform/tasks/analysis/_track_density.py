from typing import Annotated, cast

import geopandas as gpd  # type: ignore[import-untyped]
import pandas as pd
from pydantic import Field
from wt_registry import register

from ecoscope import Trajectory
from ecoscope.analysis.classifier import classify_percentile
from ecoscope.analysis.linear_time_density import calculate_ltd
from ecoscope.platform.annotations import AnyDataFrame, AnyGeoDataFrame
from ecoscope.platform.tasks.analysis._calculate_feature_density import calculate_feature_density
from ecoscope.platform.tasks.analysis._density_weighting import (
    SumWeightingSpec,
    UDWeightingSpec,
    WeightingSpec,
    normalize_density_units,
)
from ecoscope.platform.tasks.analysis._time_density import LTD_DEFAULT_PERCENTILES
from ecoscope.platform.tasks.transformation._classification import (
    DefaultLabels,
    SharedArgs,
    apply_classification,
)


def _empty_density_result(meshgrid: AnyGeoDataFrame) -> AnyGeoDataFrame:
    return gpd.GeoDataFrame(
        {"density": pd.Series(dtype="float64"), "density_bins": pd.Series(dtype="object")},
        geometry=gpd.GeoSeries(dtype="geometry"),
        crs=meshgrid.crs,
    )


def _classified_sum_density(
    geodataframe: AnyGeoDataFrame,
    meshgrid: AnyGeoDataFrame,
    weighting_spec: SumWeightingSpec,
) -> AnyGeoDataFrame:
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


def _classified_ud_density(
    geodataframe: AnyGeoDataFrame,
    meshgrid: AnyGeoDataFrame,
    weighting_spec: UDWeightingSpec,
) -> AnyGeoDataFrame:
    density_grid = calculate_ltd(traj=Trajectory(geodataframe), grid=meshgrid.copy())
    if density_grid["density"].isna().all():
        return _empty_density_result(meshgrid)

    percentiles = weighting_spec.percentiles or tuple(float(p) for p in LTD_DEFAULT_PERCENTILES)  # type: ignore[arg-type]
    result = classify_percentile(
        df=density_grid,
        percentile_levels=sorted(set(percentiles)),  # type: ignore[arg-type,misc]
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
    display value (display-unit sums for sum weightings, percentiles for UD)
    and a 'density_bins' column with ready-to-render string bin labels,
    sorted ascending with NaN cells dropped.
    """
    if geodataframe.empty:
        return _empty_density_result(meshgrid)

    if isinstance(weighting_spec, UDWeightingSpec):
        return _classified_ud_density(geodataframe, meshgrid, weighting_spec)
    return _classified_sum_density(geodataframe, meshgrid, weighting_spec)
