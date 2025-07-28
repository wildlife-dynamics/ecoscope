import geopandas as gpd
import geopandas.testing
from shapely.geometry import box
from ecoscope.base.utils import create_meshgrid
from ecoscope.analysis.UD import grid_size_from_geographic_extent
from ecoscope import Trajectory
from ecoscope.analysis.linear_time_density import calculate_ltd
from ecoscope.analysis.classifier import classify_percentile


def test_ltd_with_percentile(movebank_relocations):
    movebank_relocations.gdf = movebank_relocations.gdf[movebank_relocations.gdf["groupby_col"] == "Salif Keita"]
    traj = Trajectory.from_relocations(movebank_relocations)

    cell_size = grid_size_from_geographic_extent(traj.gdf, scale_factor=500)
    grid = create_meshgrid(
        box(*traj.gdf.total_bounds),
        in_crs=traj.gdf.crs,
        xlen=cell_size,
        ylen=cell_size,
        return_intersecting_only=False,
    )
    grid = gpd.GeoDataFrame(geometry=grid, crs=grid.crs)

    density_grid = calculate_ltd(traj=traj, grid=grid)
    density_grid = classify_percentile(
        df=density_grid,
        percentile_levels=[50, 60, 70, 80, 90, 99.9],
        input_column_name="density",
    )

    expected = gpd.read_parquet("tests/test_output/ltd.parquet")
    gpd.testing.assert_geodataframe_equal(density_grid, expected)
