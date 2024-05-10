from pathlib import Path

import pandas as pd
import pytest
import geopandas as gpd

from ecoscope.distributed.tasks import calculate_time_density


@pytest.fixture
def trajectory_gdf_parquet_path() -> str:
    return (Path(__file__).parent / "data" / "trajectory.parquet").as_posix()


@pytest.fixture
def trajectory_gdf(trajectory_gdf_parquet_path: str) -> gpd.GeoDataFrame:
    return gpd.read_parquet(trajectory_gdf_parquet_path)


def test_calculate_time_density(trajectory_gdf_parquet_path: str, trajectory_gdf: gpd.GeoDataFrame):
    raster_kws = dict(
        pixel_size=250.0,
        crs="ESRI:102022",
        nodata_value=float("nan"),
        band_count=1,
    )
    density_kws = dict(
        max_speed_factor=1.05,
        expansion_factor=1.3,
        percentiles=[50.0, 60.0, 70.0, 80.0, 90.0, 95.0],
    )
    kws = raster_kws | density_kws
    in_memory = calculate_time_density(trajectory_gdf, **kws)
    assert in_memory.shape == (6, 3)
    assert all([column in in_memory for column in ["percentile", "geometry", "area_sqkm"]])
    assert list(in_memory["area_sqkm"]) == [17.75, 13.4375, 8.875, 6.25, 4.4375, 3.125]

    # create a new version of `calculate_time_density` that is suited for distributed context
    distributed_kws = dict(
        arg_prevalidators={"trajectory_gdf": lambda path: gpd.read_parquet(path)},
        validate=True
    )
    from_parquet = calculate_time_density.replace(**distributed_kws)(trajectory_gdf_parquet_path, **kws)
    pd.testing.assert_frame_equal(in_memory, from_parquet)
    # TODO: return_postvalidator
    # return_postvalidator=persist_return_to_file
