import geopandas as gpd
from pyproj import Geod
from ecoscope import Trajectory


def calculate_ltd(
    traj: Trajectory,
    grid: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    grid = grid.reset_index().rename(columns={"index": "grid_cell_index"})
    overlay = grid.overlay(traj.gdf, how="intersection", keep_geom_type=False)
    overlay["new_distance"] = overlay["geometry"].apply(
        lambda x: Geod(ellps="WGS84").inv(*x.coords[0], *x.coords[1])[2]
    )  # Geod.inv returns tuple where the 3rd value is the distance in metres

    # new_distance is in metres so * speed_kmhr by 1000 to make speed_mhr
    # multiply this by 3600 to get time in seconds
    overlay["new_time"] = (overlay["new_distance"] / (overlay["speed_kmhr"] * 1000)) * 3600

    # This is our time density value
    grid["density"] = overlay.groupby("grid_cell_index")["new_time"].sum()

    total_time = round(grid["density"].sum(), 1)
    assert total_time == traj.gdf["timespan_seconds"].sum()

    grid["density"] = grid["density"] / total_time

    return grid
