from typing import Annotated, TypeAlias, cast

from pydantic.functional_validators import AfterValidator
from pydantic.json_schema import SkipJsonSchema
from wt_registry import register

from ecoscope.platform.annotations import AdvancedField
from ecoscope.platform.tasks.analysis._time_density import (
    CrsAnnotation,
    DEFAULT_PERCENTILES,
    ExpansionFactorAnnotation,
    TimeDensityReturnGDF,
    TrajectoryAnnotation,
    UDPercentiles,
    _coerce_percentile_strings_to_floats,
)

LocationErrorAnnotation: TypeAlias = Annotated[
    float,
    AdvancedField(
        default=20.0,
        title="GPS Location Error (meters)",
        description="Typical GPS collar accuracy - the standard deviation of a single fix's positional error.",
    ),
]
TimeStepAnnotation: TypeAlias = Annotated[
    float,
    AdvancedField(
        default=60.0,
        title="Bridge Integration Time Step (seconds)",
        description=(
            "How finely each movement segment is broken into steps when computing the"
            " density surface - smaller values are more precise but slower to compute."
        ),
    ),
]
BbmmPercentileAnnotation: TypeAlias = Annotated[
    list[UDPercentiles] | SkipJsonSchema[list[float]] | SkipJsonSchema[None],
    AdvancedField(
        default=DEFAULT_PERCENTILES,
        description="Choose the percentile levels to display.",
        title="Percentile Levels",
        json_schema_extra={"uniqueItems": True, "minItems": 1},
    ),
    AfterValidator(_coerce_percentile_strings_to_floats),
]
MaxDataGapAnnotation: TypeAlias = Annotated[
    float,
    AdvancedField(
        default=14400.0,
        title="Maximum Data Gap (seconds)",
        description=(
            "Fixes separated by a gap this long or longer are excluded rather than modeled"
            " as one highly uncertain bridge - a long gap usually reflects a data outage"
            " (e.g. a collar dropout), not real movement uncertainty."
        ),
    ),
]


@register()
def calculate_brownian_bridge_range(
    trajectory_gdf: TrajectoryAnnotation,
    crs: CrsAnnotation = "EPSG:3857",
    location_error: LocationErrorAnnotation = 20.0,
    time_step_seconds: TimeStepAnnotation = 60.0,
    expansion_factor: ExpansionFactorAnnotation = 1.3,
    percentiles: BbmmPercentileAnnotation = None,
    max_data_gap_seconds: MaxDataGapAnnotation = 14400.0,
) -> TimeDensityReturnGDF:
    """Estimate a home range using the Brownian Bridge Movement Model (BBMM)
    and return it in the same percentile/geometry/area_sqkm shape as the
    other time-density methods, so downstream map/table steps don't need to
    know which method produced it."""
    from ecoscope.analysis.percentile import get_percentile_area
    from ecoscope.analysis.UD import calculate_bbmm_range

    if percentiles is not None and len(percentiles) == 0:
        raise ValueError("Percentile values, if provided, cannot be empty.")
    percentiles = (
        sorted(set(percentiles), reverse=True)  # type: ignore[assignment]
        if percentiles is not None
        else [99.999, 95.0, 90.0, 80.0, 70.0, 60.0, 50.0]
    )

    raster_data = calculate_bbmm_range(
        trajectory_gdf,
        crs=crs,
        location_error=location_error,
        time_step_seconds=time_step_seconds,
        expansion_factor=expansion_factor,
        max_data_gap_seconds=max_data_gap_seconds,
    )
    result = get_percentile_area(percentiles, raster_data, subject_id="")  # type: ignore[arg-type]
    result = result.set_geometry("geometry", crs=raster_data.crs)
    result["area_sqkm"] = result.area / 1_000_000.0
    result = result[["percentile", "geometry", "area_sqkm"]]

    return cast(TimeDensityReturnGDF, result)
