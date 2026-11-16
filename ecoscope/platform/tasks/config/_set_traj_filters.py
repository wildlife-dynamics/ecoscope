from typing import Annotated

from pydantic import Field
from pydantic.json_schema import SkipJsonSchema
from wt_registry import register

from ecoscope.platform.annotations import AdvancedField  # type: ignore[import-untyped]
from ecoscope.platform.tasks.preprocessing._preprocessing import (  # type: ignore[import-untyped]
    TrajectorySegmentFilter,
    TrajectorySegmentFilterOverride,
    TrajectorySegmentFilterOverrides,
)
from ecoscope.platform.tasks.transformation._filtering import (  # type: ignore[import-untyped]
    BoundingBox,
    Coordinate,
)

TrajFilters = tuple[
    BoundingBox,
    list[Coordinate],
    TrajectorySegmentFilter,
    list[TrajectorySegmentFilterOverride] | None,
]


@register()
def set_traj_filters(
    bounding_box: Annotated[
        BoundingBox,
        AdvancedField(
            default=BoundingBox(),
            title="Bounding Box",
            description="Only include observations/events inside this bounding box.",
        ),
    ] = BoundingBox(),
    filter_point_coords: Annotated[
        list[Coordinate] | SkipJsonSchema[None],
        AdvancedField(
            default=[
                Coordinate(x=180, y=90),
                Coordinate(x=0, y=0),
                Coordinate(x=1, y=1),
            ],
            title="Filter Exact Point Coordinates",
            description="Exclude observations/events at these exact coordinates.",
        ),
    ] = None,
    trajectory_segment_filter: Annotated[
        TrajectorySegmentFilter,
        AdvancedField(
            default=TrajectorySegmentFilter(),
            title="Trajectory Filter",
            description="Drop trajectory segments outside these length/time/speed bounds.",
        ),
    ] = TrajectorySegmentFilter(),
    trajectory_segment_filter_overrides: Annotated[
        # `None` stays in the schema: it means "not decided", distinct from `[]`.
        TrajectorySegmentFilterOverrides | None,
        AdvancedField(
            default=None,
            title="Filter Overrides",
            description=(
                "Use a different segment filter for specific values, e.g. a subject subtype or"
                " patrol type. Values without an override use the filter above."
                " Leave empty unless the workflow supports per-type filtering."
            ),
        ),
    ] = None,
) -> TrajFilters:
    if filter_point_coords is None:
        filter_point_coords = [
            Coordinate(x=180, y=90),
            Coordinate(x=0, y=0),
            Coordinate(x=1, y=1),
        ]
    return bounding_box, filter_point_coords, trajectory_segment_filter, trajectory_segment_filter_overrides


@register()
def get_bounding_box(
    filters: Annotated[TrajFilters, Field(title="")],
) -> BoundingBox:
    return filters[0]


@register()
def get_filter_point_coords(
    filters: Annotated[TrajFilters, Field(title="")],
) -> list[Coordinate]:
    return filters[1]


@register()
def get_segment_filter(
    filters: Annotated[TrajFilters, Field(title="")],
) -> TrajectorySegmentFilter:
    return filters[2]


@register()
def get_segment_filter_overrides(
    filters: Annotated[TrajFilters, Field(title="")],
) -> list[TrajectorySegmentFilterOverride] | None:
    return filters[3]
