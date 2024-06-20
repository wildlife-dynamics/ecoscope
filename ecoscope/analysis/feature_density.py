import numpy as np


def calculate_feature_density(selection, grid, geometry_type="point"):
    def clip_density(cell):
        result = selection.clip_by_rect(*cell.bounds)
        result = result[~result.is_empty]
        if geometry_type == "point":
            return result.geometry.count()
        elif geometry_type == "line":
            return result.geometry.length.sum()
        else:
            raise ValueError("Unsupported geometry type")

    grid["density"] = grid.geometry.apply(clip_density)
    grid["density"] = grid["density"].replace(0, np.nan)  # Set 0's to nan so they don't draw on map
    return grid
