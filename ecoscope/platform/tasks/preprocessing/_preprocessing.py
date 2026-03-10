from typing import Annotated

from pydantic import BaseModel, Field, model_validator
from pydantic.json_schema import SkipJsonSchema
from wt_registry import register

from ecoscope.platform.annotations import AdvancedField
from ecoscope.platform.schemas import (
    PatrolObservationsGDF,
    SubjectGroupObservationsGDF,
    TrajectoryGDF,
)
from ecoscope.platform.tasks.transformation._filtering import Coordinate


class TrajectorySegmentFilter(BaseModel):
    min_length_meters: Annotated[
        float,
        AdvancedField(
            default=0.001,
            title="Minimum Segment Length (Meters)",
            ge=0.001,
            json_schema_extra={"minimum": 0.001},
        ),
    ] = 0.001
    max_length_meters: Annotated[
        float,
        AdvancedField(
            default=100000,
            title="Maximum Segment Length (Meters)",
            gt=0.001,
            json_schema_extra={"exclusiveMinimum": 0.001},
        ),
    ] = 100000
    min_time_secs: Annotated[
        float,
        AdvancedField(
            default=1,
            title="Minimum Segment Duration (Seconds)",
            ge=1,
            json_schema_extra={"minimum": 1},
        ),
    ] = 1
    max_time_secs: Annotated[
        float,
        AdvancedField(
            default=172800,
            title="Maximum Segment Duration (Seconds)",
            gt=1,
            json_schema_extra={"exclusiveMinimum": 1},
        ),
    ] = 172800
    min_speed_kmhr: Annotated[
        float,
        AdvancedField(
            default=0.01,
            title="Minimum Segment Speed (Kilometers per Hour)",
            gt=0.001,
            json_schema_extra={"exclusiveMinimum": 0.001},
        ),
    ] = 0.01
    max_speed_kmhr: Annotated[
        float,
        AdvancedField(
            default=500,
            title="Maximum Segment Speed (Kilometers per Hour)",
            gt=0.001,
            json_schema_extra={"exclusiveMinimum": 0.001},
        ),
    ] = 500

    @model_validator(mode="after")
    def validate_filter_values(self) -> "TrajectorySegmentFilter":
        if self.max_length_meters <= self.min_length_meters:
            raise ValueError("max_length_meters must be greater than min_length_meters")
        if self.max_time_secs <= self.min_time_secs:
            raise ValueError("max_time_secs must be greater than min_time_secs")
        if self.max_speed_kmhr <= self.min_speed_kmhr:
            raise ValueError("max_speed_kmhr must be greater than min_speed_kmhr")
        return self


@register()
def process_relocations(
    observations: PatrolObservationsGDF | SubjectGroupObservationsGDF,
    filter_point_coords: Annotated[list[Coordinate], Field()],
    relocs_columns: Annotated[
        list[str],
        Field(
            description="A list of column names to retain in the relocations dataframe.",
        ),
    ],
) -> PatrolObservationsGDF | SubjectGroupObservationsGDF:
    from ecoscope.relocations import (
        Relocations,
        RelocsCoordinateFilter,
    )

    relocs = observations if isinstance(observations, Relocations) else Relocations(gdf=observations)

    # filter relocations based on the config
    relocs.apply_reloc_filter(
        RelocsCoordinateFilter(filter_point_coords=[[coord.x, coord.y] for coord in filter_point_coords]),
        inplace=True,
    )
    relocs.remove_filtered(inplace=True)

    # subset columns
    # TODO should we remove this in favor of explicit drop columns tasks in spec?
    relocs.gdf = relocs.gdf[relocs_columns]

    # rename columns
    relocs.gdf.columns = [i.replace("extra__", "") for i in relocs.gdf.columns]
    relocs.gdf.columns = [i.replace("subject__", "") for i in relocs.gdf.columns]
    return relocs.gdf


@register()
def relocations_to_trajectory(
    relocations: PatrolObservationsGDF | SubjectGroupObservationsGDF,
    trajectory_segment_filter: Annotated[
        TrajectorySegmentFilter | SkipJsonSchema[None],
        AdvancedField(
            default=TrajectorySegmentFilter(),
            title=" ",
            description=(
                "Filter track data by setting limits on track segment length, duration, and speed."
                " Segments outside these bounds are removed, reducing noise and to focus on"
                " meaningful movement patterns."
            ),
        ),
    ] = None,
) -> TrajectoryGDF:
    from ecoscope.relocations import Relocations
    from ecoscope.trajectory import (
        Trajectory,
        TrajSegFilter,
    )

    if trajectory_segment_filter is None:
        trajectory_segment_filter = TrajectorySegmentFilter()

    # trajectory creation
    traj = Trajectory.from_relocations(Relocations(gdf=relocations))

    traj_seg_filter = TrajSegFilter(**trajectory_segment_filter.model_dump())

    # trajectory filtering
    traj.apply_traj_filter(traj_seg_filter, inplace=True)
    traj.remove_filtered(inplace=True)

    if traj.gdf.empty:
        raise ValueError("No Trajectory data left after applying segment filter")

    return traj.gdf
