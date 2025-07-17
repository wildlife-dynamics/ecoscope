import geopandas as gpd
from shapely.geometry import box
from pyproj import Geod
from ecoscope.base.utils import create_meshgrid
from ecoscope.analysis.UD import grid_size_from_geographic_extent
from ecoscope import Trajectory


def test_ltd(movebank_relocations):
    movebank_relocations.gdf = movebank_relocations.gdf[movebank_relocations.gdf["groupby_col"] == "Salif Keita"]
    traj = Trajectory.from_relocations(movebank_relocations)

    cell_size = grid_size_from_geographic_extent(traj.gdf)
    grid = create_meshgrid(
        box(*traj.gdf.total_bounds),
        in_crs=traj.gdf.crs,
        out_crs=traj.gdf.crs,
        xlen=cell_size,
        ylen=cell_size,
        return_intersecting_only=False,
    )
    grid = gpd.GeoDataFrame(geometry=grid, crs=traj.gdf.crs)
    grid = grid.reset_index().rename(columns={"index": "grid_cell_index"})
    overlay = grid.overlay(traj.gdf, how="intersection", keep_geom_type=False)

    # overlay = traj.gdf.overlay(
    #     grid,
    #     how="intersection",
    #     keep_geom_type=False
    # )
    # Interesting note that with how=intersection and keep_geom_type=false
    # grid.overlay(traj) and traj.overlay(grid) return the same result

    # divide the new dist value of the trajectory segment
    # by the speed value to recover the fractional time spent along that trajectory segment fragment.

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

    print()
