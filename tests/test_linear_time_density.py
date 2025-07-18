import geopandas as gpd
import numpy as np
from shapely.geometry import box
from ecoscope.base.utils import create_meshgrid
from ecoscope.analysis.UD import grid_size_from_geographic_extent
from ecoscope import Trajectory
from ecoscope.analysis.linear_time_density import calculate_ltd
from ecoscope.analysis.classifier import apply_color_map
from ecoscope.mapping.map import EcoMap
from ecoscope.analysis.UD import calculate_etd_range
from ecoscope.io.raster import RasterProfile
from ecoscope.analysis.percentile import get_percentile_area


def test_ltd(movebank_relocations):
    movebank_relocations.gdf = movebank_relocations.gdf[movebank_relocations.gdf["groupby_col"] == "Salif Keita"]
    traj = Trajectory.from_relocations(movebank_relocations)

    cell_size = grid_size_from_geographic_extent(traj.gdf)  # , scale_factor=500)
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
        cmap="RdYlGn",
    )

    m = EcoMap(layers=[EcoMap.polygon_layer(density_grid, fill_color_column="density_colormap")])
    m.to_html("density.html")

    percentile_levels = [50, 60, 70, 80, 90, 99.9]
    density_values = density_grid["density"].to_numpy()
    density_values = np.sort(density_values[~np.isnan(density_values)])
    csum = np.cumsum(density_values)

    percentile_values = []
    for percentile in percentile_levels:
        percentile_values.append(density_values[np.argmin(np.abs(csum[-1] * (1 - percentile / 100) - csum))])

    def find_percentile(value):
        for i in range(len(percentile_levels)):
            if value >= percentile_values[i]:
                return percentile_levels[i]
        return np.nan

    for i in range(len(percentile_levels)):
        density_grid["percentile"] = density_grid["density"].apply(find_percentile)
    filtered = density_grid[~np.isnan(density_grid["percentile"])]

    percentile_colormap = apply_color_map(
        dataframe=filtered,
        input_column_name="percentile",
        output_column_name="percentile_colormap",
        cmap="RdYlGn",
    )

    m = EcoMap(layers=[EcoMap.polygon_layer(percentile_colormap, fill_color_column="percentile_colormap")])
    m.to_html("percentiles.html")

    ##### ETD for compare
    raster_profile = RasterProfile(
        pixel_size=cell_size,
        crs="ESRI:53042",
        nodata_value="nan",
        band_count=1,
    )
    traj.gdf.sort_values("segment_start", inplace=True)

    raster_data = calculate_etd_range(
        trajectory=traj,
        max_speed_kmhr=1.05 * traj.gdf["speed_kmhr"].max(),
        raster_profile=raster_profile,
        expansion_factor=1.05,
    )

    result = get_percentile_area(
        percentile_levels=percentile_levels,
        raster_data=raster_data,
    )

    etd = apply_color_map(
        dataframe=result,
        input_column_name="percentile",
        output_column_name="percentile_colormap",
        cmap="RdYlGn",
    )

    m = EcoMap(layers=[EcoMap.polygon_layer(etd, fill_color_column="percentile_colormap")])
    m.to_html("etd.html")
