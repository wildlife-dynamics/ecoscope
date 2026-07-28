from typing import Annotated

from pydantic import Field
from pydantic.json_schema import SkipJsonSchema
from wt_registry import register

from ecoscope.platform.annotations import AdvancedField  # type: ignore[import-untyped]
from ecoscope.platform.tasks.preprocessing._preprocessing import (  # type: ignore[import-untyped]
    TrajectorySegmentFilter,
)
from ecoscope.platform.tasks.transformation._filtering import (  # type: ignore[import-untyped]
    BoundingBox,
    Coordinate,
)


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
) -> tuple[BoundingBox, list[Coordinate], TrajectorySegmentFilter]:
    if filter_point_coords is None:
        filter_point_coords = [
            Coordinate(x=180, y=90),
            Coordinate(x=0, y=0),
            Coordinate(x=1, y=1),
        ]
    return bounding_box, filter_point_coords, trajectory_segment_filter


@register()
def get_bounding_box(
    filters: Annotated[
        tuple[BoundingBox, list[Coordinate], TrajectorySegmentFilter],
        Field(title=""),
    ],
) -> BoundingBox:
    return filters[0]


@register()
def get_filter_point_coords(
    filters: Annotated[
        tuple[BoundingBox, list[Coordinate], TrajectorySegmentFilter],
        Field(title=""),
    ],
) -> list[Coordinate]:
    return filters[1]


@register()
def get_segment_filter(
    filters: Annotated[
        tuple[BoundingBox, list[Coordinate], TrajectorySegmentFilter],
        Field(title=""),
    ],
) -> TrajectorySegmentFilter:
    return filters[2]
