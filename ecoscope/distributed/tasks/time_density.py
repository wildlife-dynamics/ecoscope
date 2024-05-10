import os
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
def get_trajectories_from_earthranger(
    # client
    server,
    username,
    tcp_limit,
    sub_page_size,
    # get_subjectgroup_observations
    group_name,
    include_inactive: bool,
    since,
    until,
    # relocations filtering
    relocs_filter_coords,
    # trajectory filter
    min_length_meters: float = 0.001,
    max_length_meters: float = 10000,
    max_time_secs: float = 3600,
    min_time_secs: float = 1,
    max_speed_kmhr: float = 120,
    min_speed_kmhr: float = 0.0,    
):
    from ecoscope.base import RelocsCoordinateFilter, Relocations
    from ecoscope.io import EarthRangerIO

    earthranger_io = EarthRangerIO(
        server=server,
        username=username,
        password=os.getenv("ER_PASSWORD"),
        tcp_limit=tcp_limit,
        sub_page_size=sub_page_size,
    )
    observations = earthranger_io.get_subjectgroup_observations(
        group_name=group_name,
        include_subject_details=True,
        include_inactive=include_inactive,
        since=since,
        until=until,
    )
    # NOTE: Can possibly split this into separate task below this line, if client is no longer needed
    # -----------------------------------------------------------------------------------------------
    relocs = Relocations(observations)
    relocs.apply_reloc_filter(
        RelocsCoordinateFilter(filter_point_coords=relocs_filter_coords),
        inplace=True,
    )
    relocs.remove_filtered(inplace=True)



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
