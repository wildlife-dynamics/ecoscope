import geopandas as gpd
import numpy as np
from shapely.geometry import box
from ecoscope.base.utils import create_meshgrid
from ecoscope.analysis.UD import grid_size_from_geographic_extent
from ecoscope import Trajectory
from ecoscope.analysis.linear_time_density import calculate_ltd
from ecoscope.analysis.classifier import apply_color_map
from ecoscope.mapping.map import EcoMap


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

    density_grid = calculate_ltd(traj=traj, grid=grid)

    density_grid = apply_color_map(
        dataframe=density_grid,
        input_column_name="density",
        output_column_name="density_colormap",
        cmap="RdYlGn_r",
    )

    m = EcoMap(layers=[EcoMap.polygon_layer(density_grid, fill_color_column="density_colormap")])
    m.to_html("density.html")

    percentile_levels = [50, 60, 70, 80, 90, 99.9]
    filtered = density_grid[~np.isnan(density_grid["density"])]
    percentile_values = np.percentile(filtered["density"], percentile_levels)

    def find_percentile(value):
        for i in reversed(range(len(percentile_levels))):
            if value >= percentile_values[i]:
                return percentile_levels[i]
        return np.nan

    for i in range(len(percentile_levels)):
        filtered["percentile"] = filtered["density"].apply(find_percentile)

    percentile_colormap = apply_color_map(
        dataframe=filtered,
        input_column_name="percentile",
        output_column_name="percentile_colormap",
        cmap="RdYlGn_r",
    )

    m = EcoMap(layers=[EcoMap.polygon_layer(percentile_colormap, fill_color_column="percentile_colormap")])
    m.to_html("percentiles.html")
