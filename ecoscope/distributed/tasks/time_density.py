import tempfile
from typing import Annotated, Any

import pandera as pa
import pandas as pd
from pandera.typing import Series as PanderaSeries
from pydantic import Field

from ecoscope.distributed.decorators import distributed
from ecoscope.distributed.types import JsonSerializableDataFrameModel, DataFrame


class TrajectoryGDFSchema(JsonSerializableDataFrameModel):
    id: PanderaSeries[str] = pa.Field()
    groupby_col: PanderaSeries[str] = pa.Field()
    segment_start: PanderaSeries[pd.DatetimeTZDtype] = pa.Field(dtype_kwargs={"unit": "ns", "tz": "UTC"})
    segment_end: PanderaSeries[pd.DatetimeTZDtype] = pa.Field(dtype_kwargs={"unit": "ns", "tz": "UTC"})
    timespan_seconds: PanderaSeries[float] = pa.Field()
    dist_meters: PanderaSeries[float] = pa.Field()
    speed_kmhr: PanderaSeries[float] = pa.Field()
    heading: PanderaSeries[float] = pa.Field()
    junk_status: PanderaSeries[bool] = pa.Field()
    # pandera does support geopandas types (https://pandera.readthedocs.io/en/stable/geopandas.html)
    # but this would require this module depending on geopandas, which we are trying to avoid. so
    # unless we come up with another solution, for now we are letting `geometry` contain anything.
    geometry: PanderaSeries[Any] = pa.Field()


class TimeDensityReturnGDFSchema(JsonSerializableDataFrameModel):
    percentile: PanderaSeries[float] = pa.Field()
    geometry: PanderaSeries[Any] = pa.Field()   # see note above re: geometry typing
    area_sqkm: PanderaSeries[float] = pa.Field()


@distributed
def calculate_time_density(
    trajectory_gdf: DataFrame[TrajectoryGDFSchema],
    /,
    # raster profile
    pixel_size: Annotated[
        float,
        Field(default=250.0, description="Pixel size for raster profile."),
    ],
    crs: Annotated[str, Field(default="ESRI:102022")],
    nodata_value: Annotated[float, Field(default=float("nan"), allow_inf_nan=True)],
    band_count: Annotated[int, Field(default=1)],
    # time density
    max_speed_factor: Annotated[float, Field(default=1.05)],
    expansion_factor: Annotated[float, Field(default=1.3)],
    percentiles: Annotated[list[float], Field(default=[50.0, 60.0, 70.0, 80.0, 90.0, 95.0])],
) -> DataFrame[TimeDensityReturnGDFSchema]:
    from ecoscope.analysis.percentile import get_percentile_area
    from ecoscope.analysis.UD import calculate_etd_range
    from ecoscope.io.raster import RasterProfile

    raster_profile = RasterProfile(
        pixel_size=pixel_size,
        crs=crs,
        nodata_value=nodata_value,
        band_count=band_count,
    )
    trajectory_gdf.sort_values("segment_start", inplace=True)

    # FIXME: make `calculate_etd_range` return an in-memory raster which
    # we can pass to `get_percentile_area`, so we don't need the filesystem.
    tmp_tif_path = tempfile.NamedTemporaryFile(suffix=".tif")
    calculate_etd_range(
        trajectory_gdf=trajectory_gdf,
        output_path=tmp_tif_path,
        # Choose a value above the max recorded segment speed
        max_speed_kmhr=max_speed_factor * trajectory_gdf["speed_kmhr"].max(),
        raster_profile=raster_profile,
        expansion_factor=expansion_factor,
    )
    result = get_percentile_area(
        percentile_levels=percentiles,
        raster_path=tmp_tif_path,
    )
    result.drop(columns="subject_id", inplace=True)
    result["area_sqkm"] = result.area / 1000000.0
    return result
