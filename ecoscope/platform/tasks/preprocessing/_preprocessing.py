from collections import Counter
from typing import TYPE_CHECKING, Annotated

from pydantic import AfterValidator, BaseModel, Field, model_validator
from pydantic.json_schema import SkipJsonSchema
from wt_registry import register

from ecoscope.platform.annotations import AdvancedField, AnyGeoDataFrame
from ecoscope.platform.schemas import (
    PatrolObservationsGDF,
    SubjectGroupObservationsGDF,
    TrajectoryGDF,
)
from ecoscope.platform.tasks.transformation._filtering import Coordinate

if TYPE_CHECKING:
    from ecoscope.trajectory import TrajSegFilter


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
    # Provenance only (e.g. which named preset produced these bounds); not a user-facing
    # field, so it's excluded from the schema and from the core TrajSegFilter translation.
    preset: SkipJsonSchema[str | None] = None

    @model_validator(mode="after")
    def validate_filter_values(self) -> "TrajectorySegmentFilter":
        if self.max_length_meters <= self.min_length_meters:
            raise ValueError("max_length_meters must be greater than min_length_meters")
        if self.max_time_secs <= self.min_time_secs:
            raise ValueError("max_time_secs must be greater than min_time_secs")
        if self.max_speed_kmhr <= self.min_speed_kmhr:
            raise ValueError("max_speed_kmhr must be greater than min_speed_kmhr")
        return self

    def to_traj_seg_filter(self) -> "TrajSegFilter":
        from ecoscope.trajectory import TrajSegFilter

        return TrajSegFilter(
            min_length_meters=self.min_length_meters,
            max_length_meters=self.max_length_meters,
            min_time_secs=self.min_time_secs,
            max_time_secs=self.max_time_secs,
            min_speed_kmhr=self.min_speed_kmhr,
            max_speed_kmhr=self.max_speed_kmhr,
        )


class TrajectorySegmentFilterOverride(BaseModel):
    key: Annotated[str, Field(title="Value")]
    filter: Annotated[TrajectorySegmentFilter, Field(title="Filter")]


def _unique_override_keys(
    overrides: list[TrajectorySegmentFilterOverride],
) -> list[TrajectorySegmentFilterOverride]:
    duplicates = [key for key, count in Counter(o.key for o in overrides).items() if count > 1]
    if duplicates:
        raise ValueError(f"Duplicate trajectory segment filter override key(s): {', '.join(duplicates)}")
    return overrides


TrajectorySegmentFilterOverrides = Annotated[
    list[TrajectorySegmentFilterOverride], AfterValidator(_unique_override_keys)
]


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
    trajectory_segment_filter_overrides: Annotated[
        TrajectorySegmentFilterOverrides | SkipJsonSchema[None],
        AdvancedField(
            default=None,
            title="Filter Overrides",
            description=(
                "Use a different segment filter for specific values, e.g. a subject subtype or"
                " patrol type. Values without an override use the filter above."
            ),
        ),
    ] = None,
    # Bare `SkipJsonSchema[str | None]` crashes schema generation for a *function*
    # parameter (unlike a BaseModel field, pydantic's callable-arguments schema path
    # doesn't catch the PydanticOmit it raises); `T | SkipJsonSchema[None]` plus
    # `Field(exclude=True)` is the pattern this codebase already uses for hidden task
    # params (see `warning` on `gather_dashboard`), and it is schema-safe.
    trajectory_segment_filter_override_column: Annotated[str | SkipJsonSchema[None], Field(exclude=True)] = None,
) -> TrajectoryGDF:
    from ecoscope.relocations import Relocations
    from ecoscope.trajectory import Trajectory, traj_seg_filter_mask

    if trajectory_segment_filter is None:
        trajectory_segment_filter = TrajectorySegmentFilter()

    if trajectory_segment_filter_overrides:
        if trajectory_segment_filter_override_column is None:
            raise ValueError("trajectory_segment_filter_override_column must be set when overrides are given")
        if trajectory_segment_filter_override_column not in relocations.columns:
            raise ValueError(
                f"trajectory_segment_filter_override_column "
                f"{trajectory_segment_filter_override_column!r} not found in relocations columns"
            )

    # trajectory creation
    traj = Trajectory.from_relocations(Relocations(gdf=relocations))

    default = trajectory_segment_filter.to_traj_seg_filter()

    if trajectory_segment_filter_overrides:
        gdf = traj.gdf
        # `_create_trajsegments` renames relocation-side columns to `extra__<col>` unless
        # they already start with `extra_` (ecoscope/trajectory.py)
        traj_col = trajectory_segment_filter_override_column
        if not traj_col.startswith("extra_"):
            traj_col = f"extra__{traj_col}"
        key = gdf[traj_col]
        lookup = {o.key: o.filter.to_traj_seg_filter() for o in trajectory_segment_filter_overrides}

        junk = traj_seg_filter_mask(gdf, default) & ~key.isin(lookup)  # NaN keys fall here
        for k, f in lookup.items():
            junk |= traj_seg_filter_mask(gdf, f) & (key == k)
        gdf.loc[junk, "junk_status"] = True
    else:
        # trajectory filtering
        traj.apply_traj_filter(default, inplace=True)

    traj.remove_filtered(inplace=True)

    if traj.gdf.empty:
        raise ValueError("No Trajectory data left after applying segment filter")

    return traj.gdf


@register()
def convert_trajectory_to_relocations(trajectory_gdf: TrajectoryGDF) -> AnyGeoDataFrame:
    """Recover the point relocations a trajectory was built from.

    Some home-range methods (e.g. MCP) work directly on point fixes and
    never need a trajectory's segment/time-lag structure - this lets such
    methods reuse a trajectory that's already been fetched/filtered/grouped
    upstream, instead of re-fetching relocations through a separate pipeline
    branch.
    """
    from ecoscope.trajectory import Trajectory

    return Trajectory(gdf=trajectory_gdf).to_relocations().gdf
