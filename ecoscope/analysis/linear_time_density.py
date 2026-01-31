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

    classified_segments = traj.apply_spatial_classification(grid, output_column_name="grid_index")
    density = classified_segments.groupby("grid_index")["timespan_seconds"].sum()
    grid["density"] = grid.index.map(density)

    total_time = round(grid["density"].sum(), 1)
    grid["density"] = grid["density"] / total_time

    return grid
