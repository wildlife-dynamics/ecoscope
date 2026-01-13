import geopandas as gpd  # type: ignore[import-untyped]
from pyproj import Geod

from ecoscope import Trajectory


def calculate_ltd(
    traj: Trajectory,
    grid: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """
    Computes a density value for cells in the provided grid
    based on the total time of the trajectory segments in each cell

    Args:
    traj (ecoscope.Trajectory): The movement data.
    grid (gpd.GeoDataFrame): The meshgrid to compute density values for

    Returns
    -------
    grid : gpd.GeoDatFrame
    """
    if grid.crs.is_projected and grid.crs.axis_info[0].unit_name != "metre":
        raise ValueError("Projected grid crs must be in metres")

    classified_segments = traj.apply_spatial_classification(grid)

    ltd_grid = grid.reset_index().rename(columns={"index": "grid_cell_index"})

    density = classified_segments.groupby("spatial_index")["timespan_seconds"].sum()
    ltd_grid["density"] = ltd_grid["grid_cell_index"].map(density)

    total_time = round(ltd_grid["density"].sum(), 1)

    ltd_grid["density"] = ltd_grid["density"] / total_time
    return ltd_grid
