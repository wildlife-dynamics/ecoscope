from dataclasses import dataclass

from ecoscope.platform.annotations import AnyGeoDataFrame, DataFrame
from wt_registry import register
from wt_task import task

from ecoscope.platform.tasks.analysis._create_meshgrid import (
    AoiAnnotation,
    IntersectingOnlyAnnotation,
    create_meshgrid,
)
from ecoscope.platform.tasks.analysis._time_density import (
    AutoScaleOrCustomAnnotation,
    BandCountAnnotation,
    CrsAnnotation,
    EtdPercentileAnnotation,
    ExpansionFactorAnnotation,
    LtdPercentileAnnotation,
    MaxSpeedFactorAnnotation,
    MeshGridAnnotation,
    NoDataAnnotation,
    TimeDensityReturnGDFSchema,
    TrajectoryAnnotation,
    calculate_elliptical_time_density,
    calculate_linear_time_density,
)
from ecoscope.platform.tasks.results._ecomap import OpacityAnnotation


@dataclass
class EtdArgsWithOpacity:
    opacity: OpacityAnnotation
    auto_scale_or_custom_cell_size: AutoScaleOrCustomAnnotation
    crs: CrsAnnotation
    nodata_value: NoDataAnnotation
    band_count: BandCountAnnotation
    max_speed_factor: MaxSpeedFactorAnnotation
    expansion_factor: ExpansionFactorAnnotation
    percentiles: EtdPercentileAnnotation

    def get_etd_params(self):
        return {
            "auto_scale_or_custom_cell_size": self.auto_scale_or_custom_cell_size,
            "crs": self.crs,
            "nodata_value": self.nodata_value,
            "band_count": self.band_count,
            "max_speed_factor": self.max_speed_factor,
            "expansion_factor": self.expansion_factor,
            "percentiles": self.percentiles,
        }


@register()
def set_etd_args_with_opacity(
    opacity: OpacityAnnotation,
    auto_scale_or_custom_cell_size: AutoScaleOrCustomAnnotation = None,
    crs: CrsAnnotation = "EPSG:3857",
    nodata_value: NoDataAnnotation = "nan",
    band_count: BandCountAnnotation = 1,
    max_speed_factor: MaxSpeedFactorAnnotation = 1.05,
    expansion_factor: ExpansionFactorAnnotation = 1.3,
    percentiles: EtdPercentileAnnotation = None,
) -> EtdArgsWithOpacity:
    return EtdArgsWithOpacity(
        opacity=opacity,
        auto_scale_or_custom_cell_size=auto_scale_or_custom_cell_size,
        crs=crs,
        nodata_value=nodata_value,
        band_count=band_count,
        max_speed_factor=max_speed_factor,
        expansion_factor=expansion_factor,
        percentiles=percentiles,
    )


@dataclass
class LtdArgsWithOpacity:
    opacity: OpacityAnnotation
    auto_scale_or_custom_cell_size: AutoScaleOrCustomAnnotation
    crs: CrsAnnotation
    intersecting_only: IntersectingOnlyAnnotation
    percentiles: LtdPercentileAnnotation

    def get_meshgrid_params(self):
        return {
            "auto_scale_or_custom_cell_size": self.auto_scale_or_custom_cell_size,
            "crs": self.crs,
            "intersecting_only": self.intersecting_only,
        }

    def get_ltd_params(self):
        return {
            "percentiles": self.percentiles,
        }


@register()
def set_ltd_args_with_opacity(
    opacity: OpacityAnnotation,
    auto_scale_or_custom_cell_size: AutoScaleOrCustomAnnotation = None,
    crs: CrsAnnotation = "EPSG:3857",
    intersecting_only: IntersectingOnlyAnnotation = False,
    percentiles: LtdPercentileAnnotation = None,
) -> LtdArgsWithOpacity:
    return LtdArgsWithOpacity(
        opacity=opacity,
        auto_scale_or_custom_cell_size=auto_scale_or_custom_cell_size,
        crs=crs,
        intersecting_only=intersecting_only,
        percentiles=percentiles,
    )


@register()
def call_etd_from_combined_params(
    trajectory_gdf: TrajectoryAnnotation,
    combined_params: EtdArgsWithOpacity,
) -> DataFrame[TimeDensityReturnGDFSchema]:
    return (
        task(calculate_elliptical_time_density)
        .validate()
        .call(trajectory_gdf=trajectory_gdf, **combined_params.get_etd_params())
    )


@register()
def call_meshgrid_from_combined_params(
    aoi: AoiAnnotation,
    combined_params: LtdArgsWithOpacity,
) -> AnyGeoDataFrame:
    return (
        task(create_meshgrid)
        .validate()
        .call(aoi=aoi, **combined_params.get_meshgrid_params())
    )


@register()
def call_ltd_from_combined_params(
    trajectory_gdf: TrajectoryAnnotation,
    meshgrid: MeshGridAnnotation,
    combined_params: LtdArgsWithOpacity,
) -> AnyGeoDataFrame:
    return (
        task(calculate_linear_time_density)
        .validate()
        .call(
            trajectory_gdf=trajectory_gdf,
            meshgrid=meshgrid,
            **combined_params.get_ltd_params(),
        )
    )


@register()
def get_opacity_from_combined_params(
    combined_params: EtdArgsWithOpacity | LtdArgsWithOpacity,
) -> float:
    return combined_params.opacity
