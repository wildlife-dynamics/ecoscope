from pathlib import Path
from typing import Annotated

import pandas as pd
import pydantic
import pytest
import geopandas as gpd
from pandera.typing import DataFrame as PanderaDataFrame

from ecoscope.distributed.tasks import calculate_time_density
from ecoscope.distributed.tasks.time_density import TrajectoryGDFSchema
from ecoscope.distributed.types import DataFrame

@pytest.fixture
def trajectory_gdf_parquet_path() -> str:
    return (Path(__file__).parent / "data" / "trajectory.parquet").as_posix()


@pytest.fixture
def trajectory_gdf(trajectory_gdf_parquet_path: str) -> gpd.GeoDataFrame:
    return gpd.read_parquet(trajectory_gdf_parquet_path)


def test_trajectory_gdf_schema(trajectory_gdf: gpd.GeoDataFrame):
    class PydanticModel(pydantic.BaseModel):
        df: PanderaDataFrame[TrajectoryGDFSchema]

    PydanticModel(df=trajectory_gdf)


def test_annotated_schema(trajectory_gdf: gpd.GeoDataFrame):
    class PydanticModel(pydantic.BaseModel):
        df: DataFrame[TrajectoryGDFSchema]

    PydanticModel(df=trajectory_gdf)


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
    in_memory = calculate_time_density(trajectory_gdf, **raster_kws, **density_kws)
    assert in_memory.shape == (6, 3)
    assert all([column in in_memory for column in ["percentile", "geometry", "area_sqkm"]])
    assert list(in_memory["area_sqkm"]) == [17.75, 13.4375, 8.875, 6.25, 4.4375, 3.125]

    # now compare to distributed version
    def load_gdf_from_parquet(path: str) -> gpd.GeoDataFrame:
        return gpd.read_parquet(path)

    # create a new version of `calculate_time_density` that is suited for distributed context
    distributed_calculate_time_density = calculate_time_density.replace(
        arg_prevalidators={"trajectory_gdf": load_gdf_from_parquet},
        validate=True,
    )
    with_serialized_input_df = distributed_calculate_time_density(
        # *note* that because we passed `load_gdf_from_parquet` as a pre-validator for
        # the `trajectory_gdf` arg of `distributed_calculate_time_density`, we can now
        # pass **a string path** as the first arg to `calculate_time_density`
        trajectory_gdf_parquet_path,
        **raster_kws,
        **density_kws,
    )
    pd.testing.assert_frame_equal(in_memory, with_serialized_input_df)
