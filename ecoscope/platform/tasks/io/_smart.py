from typing import Annotated, cast

from pydantic import Field
from pydantic.json_schema import SkipJsonSchema
from wt_registry import register

from ecoscope.platform.annotations import (
    AdvancedField,
    EmptyDataFrame,
)
from ecoscope.platform.connections import SmartClient
from ecoscope.platform.schemas import EventGDF, PatrolObservationsGDF, TrajectoryGDF
from ecoscope.platform.tasks.filter._filter import TimeRange


@register(tags=["io"])
def get_patrol_observations_from_smart(
    client: SmartClient,
    time_range: Annotated[TimeRange, Field(description="Time range filter")],
    ca_uuid: Annotated[str, Field(description="Conservation Area UUID", title="Conservation Area UUID")],
    language_uuid: Annotated[str, Field(description="Language UUID", title="Language UUID")],
    patrol_mandate: Annotated[
        str | SkipJsonSchema[None],
        AdvancedField(default=None, description="Patrol Mandate", title="Patrol Mandate"),
    ] = None,
    patrol_transport: Annotated[
        str | SkipJsonSchema[None],
        AdvancedField(default=None, description="Patrol Transport", title="Patrol Transport"),
    ] = None,
) -> PatrolObservationsGDF | EmptyDataFrame:
    """Get observations for a patrol type from Smart.

    Returns point relocations for point-level use (events, maps, filtering). Do not convert
    this to a trajectory: relocations are grouped per patrol, so building a trajectory
    bridges leg-day boundaries and over-counts multi-day patrol distance/duration. Use
    ``get_patrol_trajectory_from_smart`` for patrol trajectories.
    """
    from ecoscope.relocations import Relocations

    patrol_obs_relocs = client.get_patrol_observations(
        start=time_range.since.isoformat(),
        end=time_range.until.isoformat(),
        ca_uuid=ca_uuid,
        language_uuid=language_uuid,
        patrol_mandate=patrol_mandate,
        patrol_transport=patrol_transport,
    )
    if isinstance(patrol_obs_relocs, Relocations):
        patrol_obs_relocs = patrol_obs_relocs.gdf

    return cast(PatrolObservationsGDF, patrol_obs_relocs)


@register(tags=["io"])
def get_patrol_trajectory_from_smart(
    client: SmartClient,
    time_range: Annotated[TimeRange, Field(description="Time range filter")],
    ca_uuid: Annotated[str, Field(description="Conservation Area UUID", title="Conservation Area UUID")],
    language_uuid: Annotated[str, Field(description="Language UUID", title="Language UUID")],
    patrol_mandate: Annotated[
        str | SkipJsonSchema[None],
        AdvancedField(default=None, description="Patrol Mandate", title="Patrol Mandate"),
    ] = None,
    patrol_transport: Annotated[
        str | SkipJsonSchema[None],
        AdvancedField(default=None, description="Patrol Transport", title="Patrol Transport"),
    ] = None,
) -> TrajectoryGDF | EmptyDataFrame:
    """Get one trajectory per patrol from Smart, built directly from the leg-day tracks."""
    from ecoscope.trajectory import Trajectory

    patrol_traj = client.get_patrol_trajectory(
        start=time_range.since.isoformat(),
        end=time_range.until.isoformat(),
        ca_uuid=ca_uuid,
        language_uuid=language_uuid,
        patrol_mandate=patrol_mandate,
        patrol_transport=patrol_transport,
    )
    if isinstance(patrol_traj, Trajectory):
        patrol_traj = patrol_traj.gdf

    return cast(TrajectoryGDF, patrol_traj)


@register(tags=["io"])
def get_events_from_smart(
    client: SmartClient,
    time_range: Annotated[TimeRange, Field(description="Time range filter")],
    ca_uuid: Annotated[str, Field(description="Conservation Area UUID", title="Conservation Area UUID")],
    language_uuid: Annotated[str, Field(description="Language UUID", title="Language UUID")],
) -> EventGDF | EmptyDataFrame:
    """Get events."""
    return cast(
        EventGDF,
        client.get_events(
            start=time_range.since.isoformat(),
            end=time_range.until.isoformat(),
            ca_uuid=ca_uuid,
            language_uuid=language_uuid,
        ),
    )
