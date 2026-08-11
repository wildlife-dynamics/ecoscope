from typing import Annotated, Any, cast

import geopandas as gpd  # type: ignore[import-untyped]
import pandas as pd
from pydantic import Field
from pydantic.json_schema import SkipJsonSchema
from wt_registry import register
from wt_task.skip import SkippedDependencyFallback, SkipSentinel

from ecoscope.platform.annotations import AnyDataFrame


def _drop_skip_sentinels(dfs: list[Any]) -> list[Any]:
    """Drop any SkipSentinel values from the list of dataframes."""
    return [df for df in dfs if not isinstance(df, SkipSentinel)]


@register()
def concat_dataframes(
    dfs: Annotated[
        list[AnyDataFrame],
        Field(description="DataFrames to concatenate row-wise."),
        SkippedDependencyFallback(_drop_skip_sentinels),
    ],
    ensure_columns: Annotated[
        list[str] | SkipJsonSchema[None],
        Field(
            default=[],
            description="Columns added with null values if absent after concatenation.",
        ),
    ] = None,
    reset_index: Annotated[
        bool,
        Field(description="Reset the index of the concatenated result."),
    ] = True,
) -> AnyDataFrame:
    """
    Concatenate dataframes row-wise, tolerating skipped and empty inputs.

    Args:
        dfs: DataFrames to concatenate. Skipped upstream dependencies are dropped
            before the function body runs, and empty dataframes are ignored.
        ensure_columns: Columns added with null values if absent after concatenation,
            so downstream aggregations never hit a missing column.
        reset_index: Reset the index of the concatenated result.

    Returns:
        The concatenated dataframe: a GeoDataFrame (preserving CRS) if any input
        is one, or an empty DataFrame if all inputs were dropped or empty.
    """
    ensure_columns = ensure_columns or []
    non_empty = [df for df in dfs if not df.empty]
    if not non_empty:
        return cast(AnyDataFrame, pd.DataFrame(columns=list(ensure_columns)))

    result = pd.concat(non_empty, axis=0, ignore_index=reset_index)

    geo_inputs = [df for df in non_empty if isinstance(df, gpd.GeoDataFrame)]
    if geo_inputs:
        geom_col = next((g.active_geometry_name for g in geo_inputs if g.active_geometry_name is not None), None)
        crs = next((g.crs for g in geo_inputs if g.crs is not None), None)
        if not isinstance(result, gpd.GeoDataFrame):
            result = gpd.GeoDataFrame(result)
        if geom_col is not None and geom_col in result.columns and result.active_geometry_name is None:
            result = result.set_geometry(geom_col)
        if result.active_geometry_name is not None and result.crs is None and crs is not None:
            result = result.set_crs(crs)

    for col in ensure_columns:
        if col not in result.columns:
            result[col] = pd.NA

    return cast(AnyDataFrame, result)
