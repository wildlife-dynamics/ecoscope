from dataclasses import dataclass
from typing import Annotated, Any, TypeAlias, cast

from pydantic import BaseModel, ConfigDict, Field
from wt_registry import register
from wt_task import task  # type: ignore[import-untyped]

from ecoscope.platform.annotations import AnyGeoDataFrame  # type: ignore[import-untyped]
from ecoscope.platform.tasks.analysis._bbmm import (  # type: ignore[import-untyped]
    BbmmPercentileAnnotation,
    LocationErrorAnnotation,
    MaxDataGapAnnotation,
    TimeStepAnnotation,
    calculate_brownian_bridge_range,
)
from ecoscope.platform.tasks.analysis._mcp import (  # type: ignore[import-untyped]
    McpCrsAnnotation,
    McpPercentileAnnotation,
    RelocationsAnnotation,
    calculate_minimum_convex_polygon,
)
from ecoscope.platform.tasks.analysis._time_density import (  # type: ignore[import-untyped]
    AutoScaleOrCustomAnnotation,
    CrsAnnotation,
    EtdPercentileAnnotation,
    ExpansionFactorAnnotation,
    MaxSpeedFactorAnnotation,
    TimeDensityReturnGDF,
    TrajectoryAnnotation,
    calculate_elliptical_time_density,
)
from ecoscope.platform.tasks.config._meta_tasks import (
    EtdArgsWithOpacity,  # type: ignore[import-untyped]
)
from ecoscope.platform.tasks.preprocessing._preprocessing import (  # type: ignore[import-untyped]
    TrajectoryGDF,
    convert_trajectory_to_relocations,
)

# Per-variant field wrappers below (RJSF field-blanking workaround - see
# module docstring). Each wrapper's own title is hidden via a uiSchema entry
# in spec.yaml, so the visible label lives on the inner field, unchanged
# from before wrapping. No docstring on any wrapper class deliberately:
# pydantic renders a model's docstring as its schema "description", which
# would otherwise leak this implementation note into the rendered form as
# user-facing help text.


class EtdSpeedSettings(BaseModel):
    model_config = ConfigDict(extra="ignore")
    max_speed_factor: Annotated[
        MaxSpeedFactorAnnotation,
        Field(title="Max Speed Factor (Kilometers per Hour)"),
    ]


class EtdGridCellSizeSettings(BaseModel):
    model_config = ConfigDict(extra="ignore")
    auto_scale_or_custom_cell_size: AutoScaleOrCustomAnnotation = None


class EtdExpansionFactorSettings(BaseModel):
    model_config = ConfigDict(extra="ignore")
    expansion_factor: ExpansionFactorAnnotation = 1.3


class McpShowRelocationsSettings(BaseModel):
    model_config = ConfigDict(extra="ignore")
    show_relocations: Annotated[
        bool,
        Field(title="Show Relocations"),
    ] = False


class BbmmLocationErrorSettings(BaseModel):
    model_config = ConfigDict(extra="ignore")
    location_error: Annotated[
        LocationErrorAnnotation,
        Field(title="GPS Location Error (meters)"),
    ]


class BbmmTimeStepSettings(BaseModel):
    model_config = ConfigDict(extra="ignore")
    time_step_seconds: TimeStepAnnotation = 60.0


class BbmmExpansionFactorSettings(BaseModel):
    model_config = ConfigDict(extra="ignore")
    expansion_factor: ExpansionFactorAnnotation = 1.3


class BbmmMaxDataGapSettings(BaseModel):
    model_config = ConfigDict(extra="ignore")
    max_data_gap_seconds: MaxDataGapAnnotation = 14400.0


class EtdMethodArgs(BaseModel):
    """Elliptical Time-Density. `crs`/`percentiles` are property names shared
    with `McpMethodArgs`/`BbmmMethodArgs` where the same concept applies (see
    module docstring). `speed_settings` is this variant's own genuinely-
    required disambiguator.

    Opacity is intentionally not a field here - it's a map-style concern
    (`home_range_opacity`/`set_layer_opacity` in spec.yaml), not an algorithm
    setting, so it isn't duplicated per method.
    """

    model_config = ConfigDict(extra="ignore", json_schema_extra={"title": "Elliptical Time-Density (ETD)"})
    grid_cell_size_settings: Annotated[
        EtdGridCellSizeSettings,
        Field(
            title="Grid Cell Size",
            json_schema_extra={
                "ecoscope:advanced": True,
                "default": {"auto_scale_or_custom_cell_size": {"auto_scale_or_custom": "Auto-scale"}},
            },
        ),
    ] = EtdGridCellSizeSettings()
    crs: CrsAnnotation
    speed_settings: Annotated[
        EtdSpeedSettings,
        Field(
            title="Max Speed Factor (Kilometers per Hour)",
            json_schema_extra={
                "ecoscope:advanced": True,
                "default": {"max_speed_factor": 1.05},
            },
        ),
    ]
    expansion_factor_settings: Annotated[
        EtdExpansionFactorSettings,
        Field(
            title="Shape Buffer Expansion Factor",
            json_schema_extra={
                "ecoscope:advanced": True,
                "default": {"expansion_factor": 1.3},
            },
        ),
    ] = EtdExpansionFactorSettings()
    percentiles: Annotated[EtdPercentileAnnotation, Field(description=None)]

    def get_etd_params(self) -> dict[str, Any]:
        return {
            "auto_scale_or_custom_cell_size": self.grid_cell_size_settings.auto_scale_or_custom_cell_size,
            "crs": self.crs,
            "max_speed_factor": self.speed_settings.max_speed_factor,
            "expansion_factor": self.expansion_factor_settings.expansion_factor,
            "percentiles": self.percentiles,
        }


class McpMethodArgs(BaseModel):
    """Minimum Convex Polygon. `crs`/`percentiles` are property names shared
    with `EtdMethodArgs`/`BbmmMethodArgs` (see module docstring). Has no
    field of its own that's genuinely required, so it's the implicit
    fallback variant.
    """

    model_config = ConfigDict(extra="ignore", json_schema_extra={"title": "MCP (Minimum Convex Polygon)"})
    crs: McpCrsAnnotation
    percentiles: McpPercentileAnnotation
    show_relocations_settings: Annotated[
        McpShowRelocationsSettings,
        Field(
            title="Show Relocations",
            json_schema_extra={
                "ecoscope:advanced": True,
                "default": {"show_relocations": False},
            },
        ),
    ] = McpShowRelocationsSettings()

    def get_mcp_params(self) -> dict[str, Any]:
        return {"crs": self.crs, "percentiles": self.percentiles}


class BbmmMethodArgs(BaseModel):
    """Brownian Bridge Movement Model. `crs`/`percentiles` are property names
    shared with `EtdMethodArgs`/`McpMethodArgs`, and `expansion_factor_settings`
    is additionally shared (name and shape) with `EtdMethodArgs` specifically
    (see module docstring). `location_error_settings` is this variant's own
    genuinely-required disambiguator.
    """

    model_config = ConfigDict(
        extra="ignore",
        json_schema_extra={"title": "Brownian Bridge Movement Model (BBMM)"},
    )
    crs: CrsAnnotation
    location_error_settings: Annotated[
        BbmmLocationErrorSettings,
        Field(
            title="GPS Location Error (meters)",
            json_schema_extra={
                "ecoscope:advanced": True,
                "default": {"location_error": 20.0},
            },
        ),
    ]
    time_step_settings: Annotated[
        BbmmTimeStepSettings,
        Field(
            title="Bridge Integration Time Step (seconds)",
            description=(
                "How finely each movement segment is broken into steps when computing the"
                " density surface - smaller values are more precise but slower to compute."
            ),
            json_schema_extra={"ecoscope:advanced": True},
        ),
    ] = BbmmTimeStepSettings()
    expansion_factor_settings: Annotated[
        BbmmExpansionFactorSettings,
        Field(
            title="Shape Buffer Expansion Factor",
            json_schema_extra={
                "ecoscope:advanced": True,
                "default": {"expansion_factor": 1.3},
            },
        ),
    ] = BbmmExpansionFactorSettings()
    max_data_gap_settings: Annotated[
        BbmmMaxDataGapSettings,
        Field(
            title="Maximum Data Gap (seconds)",
            description=(
                "Fixes separated by a gap this long or longer are excluded rather than modeled"
                " as one highly uncertain bridge - a long gap usually reflects a data outage"
                " (e.g. a collar dropout), not real movement uncertainty."
            ),
            json_schema_extra={"ecoscope:advanced": True},
        ),
    ] = BbmmMaxDataGapSettings()
    percentiles: BbmmPercentileAnnotation

    def get_bbmm_params(self) -> dict[str, Any]:
        return {
            "crs": self.crs,
            "location_error": self.location_error_settings.location_error,
            "time_step_seconds": self.time_step_settings.time_step_seconds,
            "expansion_factor": self.expansion_factor_settings.expansion_factor,
            "percentiles": self.percentiles,
            "max_data_gap_seconds": self.max_data_gap_settings.max_data_gap_seconds,
        }


HomeRangeMethodArgs: TypeAlias = McpMethodArgs | EtdMethodArgs | BbmmMethodArgs


@dataclass
class BbmmRasterArgs:
    """Flat, RJSF-agnostic mirror of `BbmmMethodArgs`' own fields, so
    `generate_bbmm_raster` (`ecoscope.platform.tasks.analysis._raster`)
    doesn't need to know about this module's nested wrapper models - built
    by `get_bbmm_raster_params_from_args` and passed through spec.yaml as a
    single `combined_params` value, matching `generate_etd_raster`'s own
    wiring convention (`EtdArgsWithOpacity`, defined in `_meta_tasks` for the
    same reason: consumed by a task that lives in `tasks.analysis`).
    """

    crs: CrsAnnotation
    location_error: LocationErrorAnnotation
    time_step_seconds: TimeStepAnnotation
    expansion_factor: ExpansionFactorAnnotation
    max_data_gap_seconds: MaxDataGapAnnotation

    def get_bbmm_params(self):
        return {
            "crs": self.crs,
            "location_error": self.location_error,
            "time_step_seconds": self.time_step_seconds,
            "expansion_factor": self.expansion_factor,
            "max_data_gap_seconds": self.max_data_gap_seconds,
        }


@register()
def set_home_range_args(
    args: Annotated[
        HomeRangeMethodArgs,
        Field(
            title="Method",
            default={
                "crs": "EPSG:3857",
                "percentiles": ["50", "60", "70", "80", "90", "95", "99.999"],
            },
        ),
    ],
) -> HomeRangeMethodArgs:
    return args


@register()
def call_home_range_from_args(
    trajectory_gdf: TrajectoryAnnotation,
    args: HomeRangeMethodArgs,
) -> TimeDensityReturnGDF:
    """Run whichever method `args` selected and return its result in the
    shared percentile/geometry/area_sqkm shape."""
    if isinstance(args, EtdMethodArgs):
        return (
            task(calculate_elliptical_time_density)
            .validate()
            .call(trajectory_gdf=trajectory_gdf, **args.get_etd_params())
        )
    if isinstance(args, BbmmMethodArgs):
        return (
            task(calculate_brownian_bridge_range)
            .validate()
            .call(trajectory_gdf=trajectory_gdf, **args.get_bbmm_params())
        )
    relocations_gdf = (
        task(convert_trajectory_to_relocations).validate().call(trajectory_gdf=cast(TrajectoryGDF, trajectory_gdf))
    )
    return (
        task(calculate_minimum_convex_polygon).validate().call(relocations_gdf=relocations_gdf, **args.get_mcp_params())
    )


@register()
def get_stroked_from_args(args: HomeRangeMethodArgs) -> bool:
    """MCP's few large convex hulls look clean with a stroke; ETD/BBMM's many
    small raster-cell fragments look like a messy quilted mesh when stroked
    (confirmed via direct visual comparison). Derived from the selected
    method, not user-configurable."""
    return isinstance(args, McpMethodArgs)


@register()
def get_rings_correction_from_args(args: HomeRangeMethodArgs) -> bool:
    """MCP's percentile hulls are cumulative/nested, so stacked semi-transparent
    layers compound opacity at the overlaps - always apply the rings correction
    for MCP. ETD/BBMM's percentile isopleths come from a continuous density
    surface, not nested hulls, so it never applies there. Derived from the
    selected method, not user-configurable."""
    return isinstance(args, McpMethodArgs)


@register()
def relocations_for_points_overlay(
    relocations_gdf: RelocationsAnnotation,
    args: HomeRangeMethodArgs,
) -> AnyGeoDataFrame:
    """The real relocations when MCP is selected and its own
    `show_relocations_settings.show_relocations` is enabled (the literal
    point subset its hull was computed from). An empty GeoDataFrame
    otherwise - ETD/BBMM's density surfaces have no equivalent discrete
    point subset, and an empty GeoJSON layer simply renders nothing."""
    if isinstance(args, McpMethodArgs) and args.show_relocations_settings.show_relocations:
        return relocations_gdf
    return relocations_gdf.iloc[0:0]  # type: ignore[return-value]


@register()
def any_is_non_etd_method_args(*args: Any) -> bool:
    """skipif condition: true unless every argument is an `EtdMethodArgs`
    instance. Used to skip the ETD-only raster entirely when MCP or BBMM is
    selected, rather than spending real compute on a file nobody asked
    for."""
    return any(not isinstance(a, EtdMethodArgs) for a in args)


@register()
def get_etd_raster_params_from_args(args: HomeRangeMethodArgs) -> EtdArgsWithOpacity:
    """Builds the raster task's own combined-params shape from the selected
    method's args. Only ever called when `args` is `EtdMethodArgs` - the
    corresponding spec.yaml task instance is skipped via
    `any_is_non_etd_method_args` otherwise."""
    assert isinstance(args, EtdMethodArgs)
    return EtdArgsWithOpacity(
        opacity=1.0,
        auto_scale_or_custom_cell_size=args.grid_cell_size_settings.auto_scale_or_custom_cell_size,
        crs=args.crs,
        nodata_value="nan",
        band_count=1,
        max_speed_factor=args.speed_settings.max_speed_factor,
        expansion_factor=args.expansion_factor_settings.expansion_factor,
        percentiles=args.percentiles,
    )


@register()
def any_is_non_bbmm_method_args(*args: Any) -> bool:
    """skipif condition: true unless every argument is a `BbmmMethodArgs`
    instance. Used to skip the BBMM-only raster entirely when ETD or MCP is
    selected."""
    return any(not isinstance(a, BbmmMethodArgs) for a in args)


@register()
def get_bbmm_raster_params_from_args(args: HomeRangeMethodArgs) -> BbmmRasterArgs:
    """Builds `generate_bbmm_raster`'s own flat combined-params shape from
    the selected method's args. Only ever called when `args` is
    `BbmmMethodArgs` - the corresponding spec.yaml task instance is skipped
    via `any_is_non_bbmm_method_args` otherwise."""
    assert isinstance(args, BbmmMethodArgs)
    bbmm_params = args.get_bbmm_params()
    return BbmmRasterArgs(
        crs=bbmm_params["crs"],
        location_error=bbmm_params["location_error"],
        time_step_seconds=bbmm_params["time_step_seconds"],
        expansion_factor=bbmm_params["expansion_factor"],
        max_data_gap_seconds=bbmm_params["max_data_gap_seconds"],
    )
