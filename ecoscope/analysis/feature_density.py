from typing import Callable, Literal

import geopandas as gpd  # type: ignore[import-untyped]
import numpy as np
from pandas.api.types import is_numeric_dtype


def calculate_feature_density(
    selection: gpd.GeoDataFrame,
    grid: gpd.GeoDataFrame,
    geometry_type: Literal["point", "line"] = "point",
    aggregate_function: str | Callable | None = None,
    aggregate_column: str | None = None,
) -> gpd.GeoDataFrame:
    """
    Computes the aggregate value for the selection gdf over the provided mesh grid.

    Args:
    selection (gpd.GeoDatFrame): The data.
    grid (gpd.GeoDatFrame): The mesh grid to bin the selection geometry into.
    geometry_type (Literal["point", "line"]): Whether to operate on point or line geometry.
    aggregate_function: (str | Callable | None): The aggregate function to perform on each grid cell
        ie "count", "sum" etc
        Defaults to "count" if geometry_type is "point" and "sum" if geomtery type is "line"
    aggregate_column(str | None): If provided, run the aggregate function on this column instead of the geometry.
        If not provided and geometry_type is "point", this function will count the geometry in each grid cell
        If not provided and geometry_type is "line", this function will run the aggregate over the total length
            of geometry in each grid cell.
    Returns:
    The input dataframe with a color map appended.
    """
    # project selection data to the crs of the provided grid
    selection = selection.to_crs(grid.crs)

    if aggregate_column:
        assert is_numeric_dtype(selection[aggregate_column])

    def clip_density(cell):
        if geometry_type == "point":
            cell_mask = selection.geometry.within(cell)
            result = selection[cell_mask]
            if aggregate_column:
                aggregate = aggregate_function if aggregate_function else "count"
                series_to_aggregate = result[aggregate_column]
            else:
                aggregate = "count"
                series_to_aggregate = result[cell_mask]
        elif geometry_type == "line":
            result = selection.clip_by_rect(*cell.bounds)
            result = result[~result.is_empty]
            aggregate = aggregate_function if aggregate_function else "sum"
            series_to_aggregate = result[aggregate_column] if aggregate_column else result.geometry.length
        else:
            raise ValueError("Unsupported geometry type")
        return series_to_aggregate.aggregate(aggregate)

    grid["density"] = grid.geometry.apply(clip_density)
    grid["density"] = grid["density"].replace(0, np.nan)  # Set 0's to nan so they don't draw on map
    return grid
