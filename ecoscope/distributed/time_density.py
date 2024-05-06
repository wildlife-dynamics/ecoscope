from typing import Annotated

from ecoscope.distributed.types import Field, InputDataframe, OutputDataframe

# TODO: move "Magic" types into ecoscope.distributed.types
# TODO: ENVIRONMENT + METAL
# TODO: Resources, environment, result serialization

# Backend database:
#   Workflows
#     "Time Density":
#         Tasks: [
#             "ecoscope.distributed.calculate_time_density",
#             "fifty_one_degrees.custom.func",
#           ]
# CANT EXPECT ANYTHING IN MEMORY AT CALL TIME


def calculate_time_density(
    # raster profile
    input_df: InputDataframe,
    pixel_size: Annotated[
        float,
        Field(default=250.0, description="Pixel size for raster profile."),
    ],
    crs: Annotated[str, Field(default="ESRI:102022")],
    nodata_value: Annotated[float, Field(default=float("nan"), allow_inf_nan=True)],
    band_count: Annotated[int, Field(default=1)],
    # time density
    max_speed_factor: Annotated[float, Field()],
    expansion_factor: Annotated[float, Field],
    percentiles: Annotated[list[float], Field()],
) -> OutputDataframe:
    # This is "exactly" what you would prototype in a notebook
    import geopandas as gpd

    import ecoscope
    from ecoscope.io.raster import RasterProfile

    raster_profile = RasterProfile(
        pixel_size=pixel_size,
        crs=crs,
        nodata_value=nodata_value,
        band_count=band_count,
    )
    return input_df.eco.calculate_time_density(
        max_speed_factor=max_speed_factor,
        expansion_factor=expansion_factor,
        percentiles=percentiles,
        raster_profile=raster_profile,
    )


# in airflow
from pydantic import validate_call

validate_call(calculate_time_density, validate_return=True)
