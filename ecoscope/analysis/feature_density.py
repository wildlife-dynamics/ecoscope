from typing import Literal

import geopandas as gpd  # type: ignore[import-untyped]
import numpy as np
from pandas.api.types import is_numeric_dtype


def calculate_feature_density(
    selection: gpd.GeoDataFrame,
    grid: gpd.GeoDataFrame,
    geometry_type: Literal["point", "line"] = "point",
    aggregate: str | None = None,
    aggregate_column: str | None = None,
) -> gpd.GeoDataFrame:
    # project selection data to the crs of the provided grid
    selection = selection.to_crs(grid.crs)

    if aggregate_column:
        assert is_numeric_dtype(selection[aggregate_column])

    def clip_density(cell):
        if geometry_type == "point":
            cell_mask = selection.geometry.within(cell)
            result = selection[cell_mask]
            aggregate_func = aggregate if aggregate else "count"
            series_to_aggregate = result[aggregate_column] if aggregate_column else result[cell_mask]
        elif geometry_type == "line":
            result = selection.clip_by_rect(*cell.bounds)
            result = result[~result.is_empty]
            aggregate_func = aggregate if aggregate else "sum"
            series_to_aggregate = result[aggregate_column] if aggregate_column else result.geometry.length
        else:
            raise ValueError("Unsupported geometry type")
        return series_to_aggregate.aggregate(aggregate_func)

    grid["density"] = grid.geometry.apply(clip_density)
    grid["density"] = grid["density"].replace(0, np.nan)  # Set 0's to nan so they don't draw on map
    return grid
