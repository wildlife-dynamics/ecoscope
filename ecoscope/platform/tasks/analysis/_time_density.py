import logging
from typing import Annotated, Any, Literal, TypeAlias, cast

import pandera.pandas as pa
import pandera.typing as pa_typing
from pydantic import BaseModel, ConfigDict, Field
from pydantic.functional_validators import AfterValidator
from pydantic.json_schema import SkipJsonSchema
from wt_registry import register

from ecoscope.platform.annotations import (
    AdvancedField,
    AnyGeoDataFrame,
    DataFrame,
    JsonSerializableDataFrameModel,
)

logger = logging.getLogger(__name__)


class TimeDensityReturnGDFSchema(JsonSerializableDataFrameModel):
    percentile: pa_typing.Series[float] = pa.Field()
    geometry: pa_typing.Series[Any] = pa.Field()  # see note in annotations.py
    area_sqkm: pa_typing.Series[float] = pa.Field()


class AutoScaleGridCellSize(BaseModel):
    model_config = ConfigDict(json_schema_extra={"title": "Auto-scale"})
    auto_scale_or_custom: Annotated[
        Literal["Auto-scale"],
        AdvancedField(
            default="Auto-scale",
            title="Grid Cell Size",
        ),
    ] = "Auto-scale"


class CustomGridCellSize(BaseModel):
    model_config = ConfigDict(json_schema_extra={"title": "Customize"})
    auto_scale_or_custom: Annotated[
        Literal["Customize"],
        AdvancedField(
            default="Customize",
            title="Grid Cell Size",
        ),
    ] = "Customize"
    grid_cell_size: Annotated[
        float | SkipJsonSchema[None],
        Field(
            title="Custom Grid Cell Size",
            description=(
                "Define the resolution of the raster grid (in the unit of measurement defined by the"
                " coordinate reference system set below). A smaller grid cell size provides more"
                " detail, while a larger size generalizes the data."
            ),
            gt=0,
            lt=10000,
            default=5000,
            json_schema_extra={"exclusiveMinimum": 0, "exclusiveMaximum": 10000},
        ),
    ] = 5000


def _coerce_percentile_strings_to_floats(list: list):
    """
    Utility function to coerce string representations of percentile values into floats
    """
    if list is not None:
        return [float(x) for x in list]


ETD_DEFAULT_PERCENTILES = ["50", "60", "70", "80", "90", "99.999"]
LTD_DEFAULT_PERCENTILES = ["50", "60", "70", "80", "90", "100"]
UDPercentiles = Literal["50", "60", "70", "80", "90", "95", "99", "99.999", "100"]
TrajectoryAnnotation: TypeAlias = Annotated[
    AnyGeoDataFrame,
    Field(description="The trajectory geodataframe.", exclude=True),
]
AutoScaleOrCustomAnnotation: TypeAlias = Annotated[
    AutoScaleGridCellSize | CustomGridCellSize | SkipJsonSchema[None],
    AdvancedField(
        default=None,
        json_schema_extra={
            "title": "Grid Cell Size",
            "default": {"auto_scale_or_custom": "Auto-scale"},
        },
    ),
]
CrsAnnotation: TypeAlias = Annotated[
    str,
    AdvancedField(
        default="EPSG:3857",
        title="Coordinate Reference System",
        description=(
            "The coordinate reference system in which to perform the calculation,"
            " must be a valid CRS authority code, for example ESRI:53042"
        ),
    ),
]
NoDataAnnotation: TypeAlias = Annotated[float | str, AdvancedField(default="nan")]
BandCountAnnotation: TypeAlias = Annotated[int, AdvancedField(default=1)]
MaxSpeedFactorAnnotation: TypeAlias = Annotated[
    float,
    AdvancedField(
        default=1.05,
        title="Max Speed Factor (Kilometers per Hour)",
        description=(
            "An estimate of the subject's maximum speed as a factor of the maximum measured speed value in the dataset."
        ),
    ),
]
ExpansionFactorAnnotation: TypeAlias = Annotated[
    float,
    AdvancedField(
        default=1.05,
        title="Shape Buffer Expansion Factor",
        description=(
            "Controls how far time density values spread across the grid, affecting the smoothness of the output."
        ),
    ),
]
LtdPercentileAnnotation: TypeAlias = Annotated[
    list[UDPercentiles] | SkipJsonSchema[list[float]] | SkipJsonSchema[None],
    AdvancedField(
        default=LTD_DEFAULT_PERCENTILES,
        description="Choose the time density percentile bins to display.",
        title="Percentile Levels",
        json_schema_extra={"uniqueItems": True, "minItems": 1},
    ),
    AfterValidator(_coerce_percentile_strings_to_floats),
]
EtdPercentileAnnotation: TypeAlias = Annotated[
    list[UDPercentiles] | SkipJsonSchema[list[float]] | SkipJsonSchema[None],
    AdvancedField(
        default=ETD_DEFAULT_PERCENTILES,
        description="Choose the time density percentile bins to display.",
        title="Percentile Levels",
        json_schema_extra={"uniqueItems": True, "minItems": 1},
    ),
    AfterValidator(_coerce_percentile_strings_to_floats),
]
MeshGridAnnotation: TypeAlias = Annotated[
    AnyGeoDataFrame,
    Field(
        description="The grid cells which the density is calculated from",
        exclude=True,
    ),
]


@register()
def calculate_elliptical_time_density(
    trajectory_gdf: TrajectoryAnnotation,
    auto_scale_or_custom_cell_size: AutoScaleOrCustomAnnotation = None,
    crs: CrsAnnotation = "EPSG:3857",
    nodata_value: NoDataAnnotation = "nan",
    band_count: BandCountAnnotation = 1,
    # time density
    max_speed_factor: MaxSpeedFactorAnnotation = 1.05,
    expansion_factor: ExpansionFactorAnnotation = 1.3,
    percentiles: EtdPercentileAnnotation = None,
) -> DataFrame[TimeDensityReturnGDFSchema]:
    import geopandas as gpd  # type: ignore[import-untyped]
    import pandas as pd  # type: ignore[import-untyped]

    from ecoscope.analysis.percentile import (  # type: ignore[import-untyped]
        get_percentile_area,
    )
    from ecoscope.analysis.UD import (  # type: ignore[import-untyped]
        calculate_etd_range,
        grid_size_from_geographic_extent,
    )
    from ecoscope.io.raster import RasterProfile  # type: ignore[import-untyped]
    from ecoscope.trajectory import Trajectory  # type: ignore[import-untyped]

    if percentiles is not None and len(percentiles) == 0:
        raise ValueError("Percentile values, if provided, cannot be empty.")
    percentiles = (
        sorted(list(set(percentiles)))  # type: ignore[assignment]
        if percentiles is not None
        else [50.0, 60.0, 70.0, 80.0, 90.0, 99.999]
    )

    result = pd.DataFrame(
        {
            "percentile": pd.Series(dtype="float64"),
            "geometry": gpd.GeoSeries(dtype="geometry"),
            "area_sqkm": pd.Series(dtype="float64"),
        }
    )

    if auto_scale_or_custom_cell_size is None:
        auto_scale_or_custom_cell_size = AutoScaleGridCellSize()

    if isinstance(auto_scale_or_custom_cell_size, CustomGridCellSize):
        pixel_size = auto_scale_or_custom_cell_size.grid_cell_size  # type: ignore[union-attr]
    else:
        pixel_size = grid_size_from_geographic_extent(trajectory_gdf, scale_factor=500)

    raster_profile = RasterProfile(
        pixel_size=pixel_size,
        crs=crs,
        nodata_value=nodata_value,
        band_count=band_count,
    )
    trajectory_gdf.sort_values("segment_start", inplace=True)

    raster_data = calculate_etd_range(
        trajectory=Trajectory(gdf=trajectory_gdf),
        max_speed_kmhr=max_speed_factor * trajectory_gdf["speed_kmhr"].max(),
        raster_profile=raster_profile,
        expansion_factor=expansion_factor,
    )

    if raster_data is None or raster_data.data is None or raster_data.data.size == 0:
        logger.warning("No raster data was generated.")
        return cast(DataFrame[TimeDensityReturnGDFSchema], result)

    result = get_percentile_area(
        percentile_levels=percentiles,
        raster_data=raster_data,
    )
    result.drop(columns="subject_id", inplace=True)
    result["area_sqkm"] = result.area / 1000000.0

    return cast(DataFrame[TimeDensityReturnGDFSchema], result)


@register()
def calculate_linear_time_density(
    trajectory_gdf: TrajectoryAnnotation,
    meshgrid: MeshGridAnnotation,
    percentiles: LtdPercentileAnnotation = None,
) -> AnyGeoDataFrame:
    from ecoscope import Trajectory  # type: ignore[import-untyped]
    from ecoscope.analysis.classifier import (  # type: ignore[import-untyped]
        classify_percentile,
    )
    from ecoscope.analysis.linear_time_density import (  # type: ignore[import-untyped]
        calculate_ltd,
    )

    if percentiles is not None and len(percentiles) == 0:
        raise ValueError("Percentile values, if provided, cannot be empty.")
    percentiles = (
        sorted(list(set(percentiles)))  # type: ignore[assignment]
        if percentiles is not None
        else [50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
    )

    density_grid = calculate_ltd(traj=Trajectory(trajectory_gdf), grid=meshgrid)
    result = classify_percentile(
        df=density_grid,
        percentile_levels=percentiles,
        input_column_name="density",
    )
    return cast(AnyGeoDataFrame, result)
