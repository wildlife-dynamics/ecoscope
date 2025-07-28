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
    # project traj to the crs of the provided grid
    traj.gdf = traj.gdf.to_crs(grid.crs)
    # We need the grid cell index later to count density values
    grid = grid.reset_index().rename(columns={"index": "grid_cell_index"})
    overlay = grid.overlay(traj.gdf, how="intersection", keep_geom_type=False)

    # drop anything that isn't a line
    overlay = overlay[overlay.geometry.type == "LineString"]

    overlay["fragment_distance"] = overlay["geometry"].apply(
        lambda x: Geod(ellps="WGS84").inv(*x.coords[0], *x.coords[1])[2]
    )  # Geod.inv returns tuple where the 3rd value is the distance in metres

    # fragment_distance is in metres so convert speed to 'meters per hour'
    # then multiply by 3600 to get time in seconds
    overlay["fragment_time"] = (overlay["fragment_distance"] / (overlay["speed_kmhr"] * 1000)) * 3600

    # This is our time density value
    grid["density"] = overlay.groupby("grid_cell_index")["fragment_time"].sum()

    total_time = round(grid["density"].sum(), 1)

    grid["density"] = grid["density"] / total_time
    return grid
