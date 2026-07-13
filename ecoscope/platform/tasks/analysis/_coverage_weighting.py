from typing import Annotated, Literal, TypeAlias

from pydantic import Field
from wt_registry import register

from ecoscope.platform.annotations import AnyGeoDataFrame

CoverageWeighting: TypeAlias = Literal["timespan_seconds", "dist_meters"]

WEIGHTING_DIVISORS = {"timespan_seconds": 3600.0, "dist_meters": 1000.0}
WEIGHTING_UNITS = {"timespan_seconds": "hours", "dist_meters": "km"}


@register()
def set_coverage_weighting(
    sum_column: Annotated[
        CoverageWeighting,
        Field(description="Weight each grid cell by total patrol time or distance travelled."),
    ] = "timespan_seconds",
) -> str:
    """
    Select the column used to weight patrol coverage density.
    """
    return sum_column


@register()
def normalize_coverage_density(
    df: Annotated[
        AnyGeoDataFrame,
        Field(description="Feature density output with a raw 'density' column.", exclude=True),
    ],
    sum_column: Annotated[
        CoverageWeighting,
        Field(description="The column the density was summed from; determines the display unit."),
    ],
) -> AnyGeoDataFrame:
    """
    Convert a raw density column to display units: seconds to hours, meters to kilometers.
    """
    df["density"] = df["density"] / WEIGHTING_DIVISORS[sum_column]
    return df


@register()
def get_coverage_legend_title(
    sum_column: Annotated[
        CoverageWeighting,
        Field(description="The column the density was summed from; determines the display unit."),
    ],
) -> str:
    """
    Legend title for the coverage map, including the display unit.
    """
    return f"Patrol Effort ({WEIGHTING_UNITS[sum_column]})"
