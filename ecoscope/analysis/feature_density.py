from typing import Literal
import numpy as np
import geopandas as gpd


def calculate_feature_density(
    selection: gpd.GeoDataFrame, grid: gpd.GeoDataFrame, geometry_type: Literal["point", "line"] = "point"
) -> gpd.GeoDataFrame:
    def clip_density(cell):
        if geometry_type == "point":
            result = selection.geometry.within(cell)
            result = result[result]
            return result.count()
        elif geometry_type == "line":
            result = selection.clip_by_rect(*cell.bounds)
            result = result[~result.is_empty]
            return result.geometry.length.sum()
        else:
            raise ValueError("Unsupported geometry type")

    grid["density"] = grid.geometry.apply(clip_density)
    grid["density"] = grid["density"].replace(0, np.nan)  # Set 0's to nan so they don't draw on map
    return grid
