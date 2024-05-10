from pathlib import Path

import pytest
import geopandas as gpd

from ecoscope.distributed.tasks import calculate_time_density


@pytest.fixture
def trajectory_gdf_parquet_path() -> str:
    return (Path(__file__).parent / "data" / "trajectory.parquet").as_posix()


def test_calculate_time_density(trajectory_gdf_parquet_path: str):
    trajectory_gdf = gpd.read_parquet(trajectory_gdf_parquet_path)
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
    result = calculate_time_density(trajectory_gdf, **raster_kws, **density_kws)
    assert result.shape == (6, 3)
    assert all([column in result for column in ["percentile", "geometry", "area_sqkm"]])
    assert list(result["area_sqkm"]) == [17.75, 13.4375, 8.875, 6.25, 4.4375, 3.125]
