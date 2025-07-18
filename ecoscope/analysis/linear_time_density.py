import geopandas as gpd
import numpy as np
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


def classify_percentile(
    gdf: gpd.GeoDataFrame,
    percentile_levels: list[int],
    input_column_name: str,
    output_column_name: str = "percentile",
) -> gpd.GeoDataFrame:
    input_values = gdf[input_column_name].to_numpy()
    input_values = np.sort(input_values[~np.isnan(input_values)])
    csum = np.cumsum(input_values)

    percentile_values = []
    for percentile in percentile_levels:
        percentile_values.append(input_values[np.argmin(np.abs(csum[-1] * (1 - percentile / 100) - csum))])

    def find_percentile(value):
        for i in range(len(percentile_levels)):
            if value >= percentile_values[i]:
                return percentile_levels[i]
        return np.nan

    for i in range(len(percentile_levels)):
        gdf[output_column_name] = gdf[input_column_name].apply(find_percentile)

    return gdf
