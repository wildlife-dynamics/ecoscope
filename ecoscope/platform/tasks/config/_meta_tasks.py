from dataclasses import dataclass
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field
from wt_registry import register
from wt_task import task

from ecoscope.platform.annotations import AdvancedField, AnyGeoDataFrame
from ecoscope.platform.tasks.analysis._create_meshgrid import (
    AoiAnnotation,
    IntersectingOnlyAnnotation,
    create_meshgrid,
)
from ecoscope.platform.tasks.analysis._mcp import (
    McpCrsAnnotation,
    McpPercentileAnnotation,
    RelocationsAnnotation,
    calculate_minimum_convex_polygon,
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
    TimeDensityReturnGDF,
    TrajectoryAnnotation,
    calculate_elliptical_time_density,
    calculate_linear_time_density,
)
from ecoscope.platform.tasks.preprocessing._preprocessing import convert_trajectory_to_relocations
from ecoscope.platform.tasks.results._map_utils import OpacityAnnotation


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


@dataclass
class McpArgsWithOpacity:
    opacity: OpacityAnnotation
    crs: McpCrsAnnotation
    percentiles: McpPercentileAnnotation

    def get_mcp_params(self):
        return {
            "crs": self.crs,
            "percentiles": self.percentiles,
        }


@register()
def set_mcp_args_with_opacity(
    opacity: OpacityAnnotation,
    crs: McpCrsAnnotation = "ESRI:102022",
    percentiles: McpPercentileAnnotation = None,
) -> McpArgsWithOpacity:
    return McpArgsWithOpacity(opacity=opacity, crs=crs, percentiles=percentiles)


@dataclass
class DensityGridOptions:
    opacity: float
    auto_scale_or_custom_cell_size: AutoScaleOrCustomAnnotation
    crs: CrsAnnotation
    intersecting_only: IntersectingOnlyAnnotation

    def get_meshgrid_params(self):
        return {
            "auto_scale_or_custom_cell_size": self.auto_scale_or_custom_cell_size,
            "crs": self.crs,
            "intersecting_only": self.intersecting_only,
        }


@register()
def set_density_grid_options(
    opacity: Annotated[
        float,
        AdvancedField(
            title="Heatmap Layer Opacity",
            description="Set heatmap layer transparency from 1 (fully visible) to 0 (hidden).",
            default=0.7,
            ge=0,
            le=1,
        ),
    ] = 0.7,
    auto_scale_or_custom_cell_size: AutoScaleOrCustomAnnotation = None,
    crs: CrsAnnotation = "EPSG:3857",
    intersecting_only: IntersectingOnlyAnnotation = False,
) -> DensityGridOptions:
    """
    Grid and styling options shared by gridded density heatmaps.
    """
    return DensityGridOptions(
        opacity=opacity,
        auto_scale_or_custom_cell_size=auto_scale_or_custom_cell_size,
        crs=crs,
        intersecting_only=intersecting_only,
    )


@register()
def call_etd_from_combined_params(
    trajectory_gdf: TrajectoryAnnotation,
    combined_params: EtdArgsWithOpacity,
) -> TimeDensityReturnGDF:
    return (
        task(calculate_elliptical_time_density)
        .validate()
        .call(trajectory_gdf=trajectory_gdf, **combined_params.get_etd_params())
    )


@register()
def call_meshgrid_from_combined_params(
    aoi: AoiAnnotation,
    combined_params: LtdArgsWithOpacity | DensityGridOptions,
) -> AnyGeoDataFrame:
    return task(create_meshgrid).validate().call(aoi=aoi, **combined_params.get_meshgrid_params())


@register()
def call_ltd_from_combined_params(
    trajectory_gdf: TrajectoryAnnotation,
    meshgrid: MeshGridAnnotation,
    combined_params: LtdArgsWithOpacity,
) -> TimeDensityReturnGDF:
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
def call_mcp_from_combined_params(
    relocations_gdf: RelocationsAnnotation,
    combined_params: McpArgsWithOpacity,
) -> TimeDensityReturnGDF:
    return (
        task(calculate_minimum_convex_polygon)
        .validate()
        .call(relocations_gdf=relocations_gdf, **combined_params.get_mcp_params())
    )


# ── Home Range (ETD or MCP, user-selectable) ─────────────────────────────────
#
# A single "Method" field lets the user pick ETD or MCP; only the selected
# method's own settings are present in the submitted data - EtdArgsWithOpacity/
# McpArgsWithOpacity/set_etd_args_with_opacity/set_mcp_args_with_opacity above
# remain untouched, still available for standalone ETD-only or MCP-only use.
#
# No shared discriminator field (no "method"/"type" tag) - disambiguation instead
# relies on max_speed_factor being genuinely required (no default) on EtdMethodArgs:
# MCP-submitted data never has that key, so it fails EtdMethodArgs validation and
# correctly falls through to McpMethodArgs. extra="forbid" on both models is
# required too, so ETD's full data can't *also* validate as MCP by just ignoring
# ETD's extra fields - together these make exactly one variant match any given
# payload, with nothing duplicated in the rendered form.


class EtdMethodArgs(BaseModel):
    model_config = ConfigDict(extra="forbid", json_schema_extra={"title": "Elliptical Time-Density (ETD)"})
    opacity: OpacityAnnotation
    auto_scale_or_custom_cell_size: AutoScaleOrCustomAnnotation = None
    crs: CrsAnnotation
    max_speed_factor: Annotated[
        float,
        Field(
            title="Max Speed Factor (Kilometers per Hour)",
            description=(
                "An estimate of the subject's maximum speed as a factor of the maximum"
                " measured speed value in the dataset."
            ),
            json_schema_extra={"ecoscope:advanced": True, "default": 1.05},
        ),
    ]
    expansion_factor: ExpansionFactorAnnotation = 1.3
    percentiles: EtdPercentileAnnotation

    def get_etd_params(self):
        return {
            "auto_scale_or_custom_cell_size": self.auto_scale_or_custom_cell_size,
            "crs": self.crs,
            "max_speed_factor": self.max_speed_factor,
            "expansion_factor": self.expansion_factor,
            "percentiles": self.percentiles,
        }


class McpMethodArgs(BaseModel):
    model_config = ConfigDict(extra="forbid", json_schema_extra={"title": "MCP (Minimum Convex Polygon)"})
    opacity: OpacityAnnotation
    crs: McpCrsAnnotation
    percentiles: McpPercentileAnnotation

    def get_mcp_params(self):
        return {"crs": self.crs, "percentiles": self.percentiles}


@register()
def set_home_range_args(
    args: Annotated[
        EtdMethodArgs | McpMethodArgs,
        Field(
            title="Method",
            description="The home-range estimation method to use - each option reveals its own settings below.",
            default={
                "opacity": 0.7,
                "crs": "ESRI:102022",
                "percentiles": ["50", "60", "70", "80", "90", "95", "99.999"],
            },
        ),
    ],
) -> EtdMethodArgs | McpMethodArgs:
    return args


@register()
def call_home_range_from_args(
    trajectory_gdf: TrajectoryAnnotation,
    args: EtdMethodArgs | McpMethodArgs,
) -> TimeDensityReturnGDF:
    if isinstance(args, EtdMethodArgs):
        return (
            task(calculate_elliptical_time_density)
            .validate()
            .call(trajectory_gdf=trajectory_gdf, **args.get_etd_params())
        )
    relocations_gdf = task(convert_trajectory_to_relocations).validate().call(trajectory_gdf=trajectory_gdf)
    return (
        task(calculate_minimum_convex_polygon)
        .validate()
        .call(relocations_gdf=relocations_gdf, **args.get_mcp_params())
    )


@register()
def get_stroked_from_args(args: EtdMethodArgs | McpMethodArgs) -> bool:
    """MCP's few large convex hulls look clean with a stroke; ETD's many small raster-cell
    fragments look like a messy quilted mesh when stroked (confirmed via direct visual
    comparison). So this is derived from the selected method, not user-configurable."""
    return isinstance(args, McpMethodArgs)


@register()
def relocations_for_points_overlay(
    relocations_gdf: RelocationsAnnotation,
    args: EtdMethodArgs | McpMethodArgs,
) -> AnyGeoDataFrame:
    """Real relocations when MCP is selected (the literal point subset its hull was computed
    from - directly informative to show). An empty GeoDataFrame when ETD is selected (a
    density surface has no equivalent discrete point subset, and an empty GeoJSON layer
    renders nothing)."""
    if isinstance(args, McpMethodArgs):
        return relocations_gdf
    return relocations_gdf.iloc[0:0]  # type: ignore[return-value]


@register()
def get_home_range_opacity(args: EtdMethodArgs | McpMethodArgs) -> float:
    return args.opacity


@register()
def any_is_mcp_method_args(*args: Any) -> bool:
    """skipif condition: true if any argument is a McpMethodArgs instance. Used to skip
    the ETD-only raster entirely when MCP is selected, rather than silently falling back
    to ETD's own defaults and spending real compute on a file nobody configured or asked
    for (MCP has no raster/UD surface of its own)."""
    return any(isinstance(a, McpMethodArgs) for a in args)


@register()
def get_etd_raster_params_from_args(args: EtdMethodArgs | McpMethodArgs) -> EtdArgsWithOpacity:
    """The GeoTIFF raster only makes sense for ETD (MCP has no raster/UD surface) - the
    spec.yaml task instance for this is skipped entirely via any_is_mcp_method_args when
    MCP is selected, so `args` is always EtdMethodArgs by the time this actually runs."""
    assert isinstance(args, EtdMethodArgs)
    return EtdArgsWithOpacity(
        opacity=1.0,
        auto_scale_or_custom_cell_size=args.auto_scale_or_custom_cell_size,
        crs=args.crs,
        nodata_value="nan",
        band_count=1,
        max_speed_factor=args.max_speed_factor,
        expansion_factor=args.expansion_factor,
        percentiles=args.percentiles,
    )


@register()
def get_opacity_from_combined_params(
    combined_params: EtdArgsWithOpacity | LtdArgsWithOpacity | DensityGridOptions | McpArgsWithOpacity,
) -> float:
    return combined_params.opacity
