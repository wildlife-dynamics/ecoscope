import geopandas as gpd
import numpy as np
from shapely.strtree import STRtree


def calculate_feature_density(selection, grid, geometry_type="point"):
    tree = STRtree(selection.geometry)

    def clip_density(cell):
        result = tree.query(cell)
        if len(result) == 0:
            return 0
        result_gdf = gpd.GeoDataFrame(geometry=selection.iloc[result].geometry, crs=selection.crs)
        if geometry_type == "point":
            return len(result_gdf)
        elif geometry_type == "line":
            return result_gdf.length.sum() / 1000  # Convert to kilometers
        else:
            raise ValueError("Unsupported geometry type")

    grid["density"] = grid.geometry.apply(clip_density)
    grid["density"] = grid["density"].replace(0, np.nan)  # Set 0's to nan so they don't draw on map
    return grid
