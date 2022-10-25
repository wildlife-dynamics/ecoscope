from math import ceil, floor
import numpy as np

import ecoscope

def get_consecutive_items_number(idxs):
    gaps = [[s, e] for s, e in zip(idxs, idxs[1:]) if s+1 < e]
    edges = iter(idxs[:1] + sum(gaps, []) + idxs[-1:])
    return len(list(zip(edges, edges)))

def get_recursions(relocations, resolution):
    relocations = relocations.reset_index(drop=True)
    if not relocations["fixtime"].is_monotonic:
            relocations.sort_values("fixtime", inplace=True)

    diameter = ceil(resolution)*2
    utm_crs = relocations.estimate_utm_crs()
    relocations.to_crs(utm_crs, inplace=True)

    geom = relocations["geometry"]
    eastings = np.array([geom.iloc[i].x for i in range(len(geom))]).flatten()
    northings = np.array([geom.iloc[i].y for i in range(len(geom))]).flatten()

    grid = ecoscope.base.Grid(eastings, northings, diameter)
    
    mosaic = np.full((grid.n_cols, grid.n_rows), np.nan)
    grid_cells_dict = {}
    for i in range(len(geom)):
        point = geom.iloc[i]
        grid_cell = grid.inverse_transform * (point.x, point.y)
        row, col = floor(grid_cell[0]), ceil(grid_cell[1])
        if (col, row) in grid_cells_dict:
            grid_cells_dict[(col, row)].append(i)
        else:
            grid_cells_dict[(col, row)] = [i]
    
    for grid_cell in grid_cells_dict:
        grid_cells_dict[grid_cell] = get_consecutive_items_number(grid_cells_dict[grid_cell])
        
    recursion_values = []
    for i in range(len(geom)):
        point = geom.iloc[i]
        grid_cell = grid.inverse_transform * (point.x, point.y)
        row, col = floor(grid_cell[0]), ceil(grid_cell[1])
        recursion_values.append(grid_cells_dict[(col, row)])
    
    return recursion_values