import logging
import os
from dataclasses import dataclass
from typing import Annotated, Literal, cast

import pandas as pd
from pydantic import AfterValidator, Field, SecretStr
from pydantic.json_schema import SkipJsonSchema
from tqdm import tqdm
from wt_registry import register
from wt_task import task
from wt_task.skip import SkippedDependencyFallback

from ecoscope.platform.annotations import AdvancedField, AnyDataFrame, EmptyDataFrame
from ecoscope.platform.connections import EarthRangerClient
from ecoscope.platform.indexes import CompositeFilter
from ecoscope.platform.schemas import (
    EventGDF,
    EventsWithDisplayNamesGDF,
    PatrolObservationsGDF,
    PatrolsDF,
    RegionsGDF,
    SpatialFeaturesGroup,
    SubjectGroupObservationsGDF,
)
from ecoscope.platform.serde import _persist_bytes
from ecoscope.platform.tasks.filter._filter import TimeRange
from ecoscope.platform.tasks.io._persist_df_wrapper import (
    _fallback_to_empty_grouped,
    _hash_grouper_key,
)

logger = logging.getLogger(__name__)


def _make_warehouse_client_from_env(er_site_url: str, er_api_token: SecretStr | None):
    """Create an ERWarehouseClient if warehouse env vars are configured.

    Returns None when the warehouse is not enabled, causing callers to
    fall back to the legacy EarthRanger API client.
    """
    if os.environ.get("USE_EARTHRANGER_WAREHOUSE_API", "false").lower() == "true" and er_api_token is not None:
        from ecoscope_earthranger_io_core.client import ERWarehouseClient  # type: ignore[import-untyped]

        warehouse_api_base_url = os.environ.get("EARTHRANGER_WAREHOUSE_API_BASE_URL")
        logger.debug("Using ERWarehouseClient (warehouse_base_url=%s)", warehouse_api_base_url)
        return ERWarehouseClient(
            server=er_site_url,
            token=er_api_token,
            warehouse_base_url=warehouse_api_base_url,
        )
    return None


def _strip_whitespace_from_list_items(v: list[str]):
    return [item.strip() for item in v]


@register()
def set_patrol_types(
    patrol_types: Annotated[
        list[str],
        Field(description="Specify the patrol type(s) to analyze (optional). Leave empty to analyze all patrol types."),
    ],
) -> Annotated[
    list[str],
    Field(description="Passthrough selected patrol types for use in downstream EarthRanger queries"),
]:
    return patrol_types


PatrolStatus = Literal["active", "overdue", "done", "cancelled"]
AppendCategorySelection = Literal["duplicates", "always", "never"]

ExclusionFilter = Literal[
    "none",  # no filtering: returns everything
    "clean",  # exclusion flag 0: passes back clean data
    "manually_filtered",  # exclusion flag 1: passes back manually filtered data
    "automatically_filtered",  # exclusion flag 2: passes back automatically filtered data
    "manually_and_automatically_filtered",  # exclusion flag 3: passes back both manual and automatically filtered data
]
_EXCLUSION_FILTER_TO_INT: dict[str, int | None] = {
    "none": None,
    "clean": 0,
    "manually_filtered": 1,
    "automatically_filtered": 2,
    "manually_and_automatically_filtered": 3,
}
PatrolStatusField = AdvancedField(
    default=["done"],
    title="Patrol Status",
    description=(
        "Choose to analyze patrols with a certain status. If left empty, patrols of all status will be analyzed"
    ),
    json_schema_extra={"uniqueItems": True},
)
PatrolStatusAnnotation = Annotated[list[PatrolStatus] | SkipJsonSchema[None], PatrolStatusField]
AppendCategorySelectionAnnotation = Annotated[AppendCategorySelection, AdvancedField(default="duplicates")]
TimeRangeAnnotation = Annotated[TimeRange, Field(description="Time range filter")]
PatrolTypesAnnotation = Annotated[
    list[str],
    AfterValidator(_strip_whitespace_from_list_items),
    Field(description="Specify the patrol type(s) to analyze (optional). Leave empty to analyze all patrol types."),
]
EventTypesAnnotation = Annotated[
    list[str],
    AfterValidator(_strip_whitespace_from_list_items),
    Field(
        description=(
            "Specify the event type(s) to analyze (optional). Leave this section empty to analyze all event types."
        ),
        title="Event Types",
    ),
]
IncludePatrolDetailsAnnotation = Annotated[bool, AdvancedField(default=True, description="Include patrol details")]
RaiseOnEmptyAnnotation = Annotated[
    bool,
    AdvancedField(
        default=False,
        description="Whether or not to abort the workflow if no data is returned from EarthRanger",
    ),
]
IncludeNullGeometryAnnotation = Annotated[
    bool,
    Field(
        default=True,
        title="Include Events Without a Geometry (point or polygon)",
    ),
]
TruncateToTimeRangeAnnotation = Annotated[
    bool,
    AdvancedField(
        default=True,
        description="""\
        Whether to truncate events to the time range. Defaults to True. If False,
        events outside the time range will be included, which may happen because
        the time range filters the query for `patrol_type`, but there may be events
        associated with those patrol types that are outside the time range. If False
        and events outside the time range are included, downstream processing may be
        affected.
        """,
    ),
]
SubPageSizeAnnotation = Annotated[
    int | SkipJsonSchema[None],
    AdvancedField(
        default=None,
        description="""\
        Manually set the page size for underlying ER API requests.
        If left as None, this will use the underlying client default value (4000)
        """,
    ),
]
IncludeDisplayValuesAnnotation = Annotated[
    bool,
    AdvancedField(
        default=True,
        description="Whether or not to include display values for event types",
    ),
]
PatrolsOverlapDateRangeAnnotation = Annotated[
    bool,
    AdvancedField(
        default=True,
        description="Whether or not to include patrols that start or end outside of the time range",
    ),
]


EventColumns = Literal[
    "id",
    "location",
    "time",
    "end_time",
    "message",
    "provenance",
    "event_type",
    "event_category",
    "priority",
    "priority_label",
    "attributes",
    "comment",
    "title",
    "reported_by",
    "state",
    "is_contained_in",
    "sort_at",
    "patrol_segments",
    "geometry",
    "updated_at",
    "created_at",
    "icon_id",
    "serial_number",
    "url",
    "image_url",
    "geojson",
    "is_collection",
    "event_details",
    "related_subjects",
    "patrols",
]
DefaultEventColumns: list[EventColumns] = [
    "id",
    "time",
    "event_type",
    "event_category",
    "title",
    "reported_by",
    "created_at",
    "serial_number",
    "is_collection",
    "event_details",
    "geometry",
]
EventColumnsAnnotation = Annotated[
    list[EventColumns] | SkipJsonSchema[None],
    Field(
        default=DefaultEventColumns,
        title="Event Columns",
        description="Choose the interested event columns. If none is chosen, all columns will be returned.",
        json_schema_extra={"uniqueItems": True},
    ),
]
IncludeDetailsAnnotation = Annotated[
    bool,
    AdvancedField(
        default=False,
        description="Whether or not to include event details",
    ),
]
IncludeUpdatesAnnotation = Annotated[
    bool,
    AdvancedField(
        default=False,
        description="Whether or not to include event updates",
    ),
]
IncludeRelatedEventsAnnotation = Annotated[
    bool,
    AdvancedField(
        default=False,
        description="Whether or not to include related events",
    ),
]
ForcePointGeometryAnnotation = Annotated[
    bool,
    AdvancedField(
        default=True,
        description=(
            "If True, polygon/multipolygon event geometries are reduced to their centroid."
            " Set False to preserve native geometry types."
        ),
    ),
]
AnalysisFieldAnnotation = Annotated[
    str,
    Field(
        title="Analysis Field - Numeric field to analyze",
        description=(
            "Select the numeric field the work will use for calculations,"
            " i.e. the sum of the field from all events in the workflow."
        ),
    ),
]
AnalysisFieldLabelAnnotation = Annotated[
    str,
    Field(
        title="Label for analysis data",
        description="The provided text will be used to label the analysis data in the outputs.",
    ),
]
AnalysisFieldUnitAnnotation = Annotated[
    str,
    Field(
        title="Units - What does the number represent?",
        description='Describe the unit or entity for the number,e.g., "elephants", "meters", "snares", "$USD."',
    ),
]
CategoryFieldAnnotation = Annotated[
    str,
    Field(
        title="Category Field - Choice field to categorize by",
        description=(
            "Differentiate events with the choices in a choice field."
            " Each choice will be represented by a different color in the outputs."
        ),
    ),
]
CategoryFieldLabelAnnotation = Annotated[
    str,
    Field(
        title="Label for category data",
        description="The provided text will be used to label the category data in the outputs.",
        default="",
    ),
]
SingleEventTypeAnnotation = Annotated[str, Field(title="Events Type")]


@dataclass
class CombinedPatrolAndEventsParams:
    client: str
    time_range: TimeRangeAnnotation
    patrol_types: PatrolTypesAnnotation
    event_types: EventTypesAnnotation
    status: PatrolStatusAnnotation | None = None
    include_patrol_details: IncludePatrolDetailsAnnotation = True
    raise_on_empty: RaiseOnEmptyAnnotation = True
    include_null_geometry: IncludeNullGeometryAnnotation = True
    truncate_to_time_range: TruncateToTimeRangeAnnotation = True
    sub_page_size: SubPageSizeAnnotation = 100
    patrols_overlap_daterange: PatrolsOverlapDateRangeAnnotation = True

    def get_patrol_observations_params(self):
        return {
            "client": self.client,
            "time_range": self.time_range,
            "patrol_types": self.patrol_types,
            "status": self.status,
            "include_patrol_details": self.include_patrol_details,
            "raise_on_empty": self.raise_on_empty,
            "sub_page_size": self.sub_page_size,
            "patrols_overlap_daterange": self.patrols_overlap_daterange,
        }

    def get_patrol_events_params(self):
        return {
            "client": self.client,
            "time_range": self.time_range,
            "patrol_types": self.patrol_types,
            "event_types": self.event_types,
            "status": self.status,
            "include_null_geometry": self.include_null_geometry,
            "truncate_to_time_range": self.truncate_to_time_range,
            "raise_on_empty": self.raise_on_empty,
            "sub_page_size": self.sub_page_size,
            "patrols_overlap_daterange": self.patrols_overlap_daterange,
        }

    def get_patrols_params(self):
        return {
            "client": self.client,
            "time_range": self.time_range,
            "patrol_types": self.patrol_types,
            "status": self.status,
            "raise_on_empty": self.raise_on_empty,
            "sub_page_size": self.sub_page_size,
            "patrols_overlap_daterange": self.patrols_overlap_daterange,
        }

    def get_patrol_observations_from_patrols_df_params(self):
        return {
            "client": self.client,
            "include_patrol_details": self.include_patrol_details,
            "raise_on_empty": self.raise_on_empty,
            "sub_page_size": self.sub_page_size,
        }

    def unpack_patrol_events_params(self):
        return {
            "event_types": self.event_types,
            "time_range": self.time_range,
            "include_null_geometry": self.include_null_geometry,
            "truncate_to_time_range": self.truncate_to_time_range,
            "raise_on_empty": self.raise_on_empty,
        }


@register()
def set_patrol_status(
    status: PatrolStatusAnnotation = None,
) -> PatrolStatusAnnotation:
    if status is None:
        status = ["done"]
    return status


@register()
def set_patrols_and_patrol_events_params(
    client: str,
    time_range: TimeRangeAnnotation,
    patrol_types: PatrolTypesAnnotation,
    event_types: EventTypesAnnotation,
    status: PatrolStatusAnnotation = None,
    include_patrol_details: IncludePatrolDetailsAnnotation = True,
    raise_on_empty: RaiseOnEmptyAnnotation = True,
    include_null_geometry: IncludeNullGeometryAnnotation = True,
    truncate_to_time_range: TruncateToTimeRangeAnnotation = True,
    sub_page_size: SubPageSizeAnnotation = 100,
    patrols_overlap_daterange: PatrolsOverlapDateRangeAnnotation = True,
) -> Annotated[
    CombinedPatrolAndEventsParams,
    Field(description="Passthrough selected patrol and event types for use in downstream EarthRanger queries"),
]:
    return CombinedPatrolAndEventsParams(
        client=client,
        time_range=time_range,
        patrol_types=patrol_types,
        event_types=event_types,
        status=status,
        include_patrol_details=include_patrol_details,
        raise_on_empty=raise_on_empty,
        include_null_geometry=include_null_geometry,
        truncate_to_time_range=truncate_to_time_range,
        sub_page_size=sub_page_size,
        patrols_overlap_daterange=patrols_overlap_daterange,
    )


@register(tags=["io"])
def get_subjectgroup_observations(
    client: EarthRangerClient,
    subject_group_name: Annotated[
        str,
        Field(description="⚠️ The use of a group with mixed subtypes could lead to unexpected results"),
    ],
    time_range: Annotated[TimeRange, Field(description="Time range filter")],
    raise_on_empty: Annotated[
        bool,
        AdvancedField(
            default=True,
            description="Whether or not to abort the workflow if no data is returned from EarthRanger",
        ),
    ] = True,
    include_details: Annotated[
        bool,
        AdvancedField(
            default=False,
            title="Include Observation Details",
            description="Whether or not to include observation details",
        ),
    ] = False,
    include_subjectsource_details: Annotated[
        bool,
        AdvancedField(
            default=False,
            title="Include Subject Source Details",
            description="Whether or not to include subject source details",
        ),
    ] = False,
    filter: Annotated[
        ExclusionFilter,
        AdvancedField(
            default="clean",
            description="Filter observations based on exclusion flags.",
        ),
    ] = "clean",
) -> SubjectGroupObservationsGDF | EmptyDataFrame:
    """Get observations for a subject group from EarthRanger."""
    from ecoscope.relocations import Relocations

    filter_int = _EXCLUSION_FILTER_TO_INT[filter]

    if warehouse_client := _make_warehouse_client_from_env(
        er_site_url=client.server,
        er_api_token=SecretStr(client.token) if client.token else None,
    ):
        import geopandas as gpd  # type: ignore[import-untyped]

        table = warehouse_client.get_subjectgroup_observations(
            subject_group_name=subject_group_name,
            include_subject_details=True,
            include_inactive=True,
            include_details=include_details,
            include_subjectsource_details=include_subjectsource_details,
            since=time_range.since.isoformat(),
            until=time_range.until.isoformat(),
            filter=filter_int,
        )
        subject_group_obs_relocs = gpd.GeoDataFrame.from_arrow(table)
    else:
        subject_group_obs_relocs = client.get_subjectgroup_observations(
            subject_group_name=subject_group_name,
            include_subject_details=True,
            include_inactive=True,
            include_details=include_details,
            include_subjectsource_details=include_subjectsource_details,
            since=time_range.since.isoformat(),
            until=time_range.until.isoformat(),
            filter=filter_int,
        )
        if isinstance(subject_group_obs_relocs, Relocations):
            subject_group_obs_relocs = subject_group_obs_relocs.gdf

    if raise_on_empty and subject_group_obs_relocs.empty:
        raise ValueError("No data returned from EarthRanger for get_subjectgroup_observations")

    return cast(SubjectGroupObservationsGDF, subject_group_obs_relocs)


@register(tags=["io"])
def get_patrol_observations(
    client: EarthRangerClient,
    time_range: TimeRangeAnnotation,
    patrol_types: PatrolTypesAnnotation,
    status: PatrolStatusAnnotation = None,
    include_patrol_details: IncludePatrolDetailsAnnotation = True,
    raise_on_empty: RaiseOnEmptyAnnotation = True,
    sub_page_size: SubPageSizeAnnotation = 100,
    patrols_overlap_daterange: PatrolsOverlapDateRangeAnnotation = True,
) -> PatrolObservationsGDF | EmptyDataFrame:
    """Get observations for a patrol type from EarthRanger."""
    from ecoscope.relocations import Relocations

    if status is None:
        status = ["done"]

    if warehouse_client := _make_warehouse_client_from_env(
        er_site_url=client.server,
        er_api_token=SecretStr(client.token) if client.token else None,
    ):
        import geopandas as gpd  # type: ignore[import-untyped]

        table = warehouse_client.get_patrol_observations_with_patrol_filter(
            since=time_range.since.isoformat(),
            until=time_range.until.isoformat(),
            patrol_type_value=patrol_types,
            status=status,
            include_patrol_details=include_patrol_details,
            sub_page_size=sub_page_size,
            patrols_overlap_daterange=patrols_overlap_daterange,
        )
        patrol_obs_relocs = gpd.GeoDataFrame.from_arrow(table)
    else:
        patrol_obs_relocs = client.get_patrol_observations_with_patrol_filter(
            since=time_range.since.isoformat(),
            until=time_range.until.isoformat(),
            patrol_type_value=patrol_types,
            status=status,
            include_patrol_details=include_patrol_details,
            sub_page_size=sub_page_size,
            patrols_overlap_daterange=patrols_overlap_daterange,
        )
        if isinstance(patrol_obs_relocs, Relocations):
            patrol_obs_relocs = patrol_obs_relocs.gdf

    if raise_on_empty and patrol_obs_relocs.empty:
        raise ValueError("No data returned from EarthRanger for get_patrol_observations_with_patrol_filter")

    return cast(PatrolObservationsGDF, patrol_obs_relocs)


@register(tags=["io"])
def get_patrol_events(
    client: EarthRangerClient,
    time_range: TimeRangeAnnotation,
    patrol_types: PatrolTypesAnnotation,
    event_types: EventTypesAnnotation,
    status: PatrolStatusAnnotation = None,
    include_null_geometry: IncludeNullGeometryAnnotation = True,
    truncate_to_time_range: TruncateToTimeRangeAnnotation = True,
    raise_on_empty: RaiseOnEmptyAnnotation = True,
    sub_page_size: SubPageSizeAnnotation = 100,
    include_display_values: IncludeDisplayValuesAnnotation = False,
    patrols_overlap_daterange: PatrolsOverlapDateRangeAnnotation = True,
) -> EventGDF | EventsWithDisplayNamesGDF | EmptyDataFrame:
    """Get events from patrols."""

    if status is None:
        status = ["done"]

    events = client.get_patrol_events(
        since=time_range.since.isoformat(),
        until=time_range.until.isoformat(),
        patrol_type_value=patrol_types,
        event_type=event_types,
        status=status,
        drop_null_geometry=not include_null_geometry,
        sub_page_size=sub_page_size,
        patrols_overlap_daterange=patrols_overlap_daterange,
    )

    if raise_on_empty and events.empty:
        raise ValueError("No data returned from EarthRanger for get_patrol_events")

    if truncate_to_time_range and not events.empty:
        events = events.loc[  # type: ignore[assignment]
            (events.time >= time_range.since) & (events.time <= time_range.until)
        ]

    if not events.empty and include_display_values:
        events = client.get_event_type_display_names_from_events(events, append_category_names="duplicates")

    return cast(EventGDF | EventsWithDisplayNamesGDF, events)


@register(tags=["io"])
def get_events(
    client: EarthRangerClient,
    time_range: TimeRangeAnnotation,
    event_types: EventTypesAnnotation,
    event_columns: EventColumnsAnnotation = None,
    include_null_geometry: IncludeNullGeometryAnnotation = True,
    raise_on_empty: RaiseOnEmptyAnnotation = True,
    include_details: IncludeDetailsAnnotation = False,
    include_updates: IncludeUpdatesAnnotation = False,
    include_related_events: IncludeRelatedEventsAnnotation = False,
    include_display_values: IncludeDisplayValuesAnnotation = False,
    force_point_geometry: ForcePointGeometryAnnotation = True,
) -> EventGDF | EventsWithDisplayNamesGDF | EmptyDataFrame:
    """Get events."""
    event_type_ids: list[str] = []
    no_ids_found = False
    # Resolve event_type ids from the values input in event_types
    # If none are resolved we flag this explicitly in `no_ids_found`
    # as we need to treat this case separately from empty input
    if len(event_types) > 0:
        all_event_types = pd.DataFrame(client.get_event_types())
        event_type_ids = all_event_types[all_event_types["value"].isin(event_types)]["id"].to_list()
        no_ids_found = not event_type_ids

    events_df = (
        pd.DataFrame()
        if no_ids_found
        else client.get_events(
            since=time_range.since.isoformat(),
            until=time_range.until.isoformat(),
            event_type=event_type_ids,
            drop_null_geometry=not include_null_geometry,
            include_details=include_details,
            include_updates=include_updates,
            include_related_events=include_related_events,
            force_point_geometry=force_point_geometry,
        )
    )

    if raise_on_empty and events_df.empty:
        raise ValueError("No data returned from EarthRanger for get_events")

    if not events_df.empty:
        events_df = events_df.reset_index()

        if event_columns is not None:
            events_df = events_df[event_columns]  # type: ignore[assignment]

        if include_display_values:
            events_df = client.get_event_type_display_names_from_events(
                events_df,
                append_category_names="duplicates",
            )

    return cast(
        EventGDF | EventsWithDisplayNamesGDF,
        events_df,
    )


@register(tags=["io"])
def get_patrol_observations_from_combined_params(
    combined_params: CombinedPatrolAndEventsParams,
) -> PatrolObservationsGDF | EmptyDataFrame:
    return task(get_patrol_observations).validate().call(**combined_params.get_patrol_observations_params())


@register(tags=["io"])
def get_patrol_events_from_combined_params(
    combined_params: CombinedPatrolAndEventsParams,
) -> EventGDF | EventsWithDisplayNamesGDF | EmptyDataFrame:
    return task(get_patrol_events).validate().call(**combined_params.get_patrol_events_params())


@register(tags=["io"])
def get_patrols_from_combined_params(
    combined_params: CombinedPatrolAndEventsParams,
) -> PatrolsDF | EmptyDataFrame:
    return task(get_patrols).validate().call(**combined_params.get_patrols_params())


@register(tags=["io"])
def get_patrol_observations_from_patrols_df_and_combined_params(
    patrols_df: PatrolsDF,
    combined_params: CombinedPatrolAndEventsParams,
) -> PatrolObservationsGDF | EmptyDataFrame:
    return (
        task(get_patrol_observations_from_patrols_df)
        .validate()
        .call(
            patrols_df=patrols_df,
            **combined_params.get_patrol_observations_from_patrols_df_params(),
        )
    )


@register()
def unpack_events_from_patrols_df_and_combined_params(
    patrols_df: PatrolsDF,
    combined_params: CombinedPatrolAndEventsParams,
) -> EventGDF | EmptyDataFrame:
    return (
        task(unpack_events_from_patrols_df)
        .validate()
        .call(patrols_df=patrols_df, **combined_params.unpack_patrol_events_params())
    )


@register(tags=["io"])
def get_patrols(
    client: EarthRangerClient,
    time_range: TimeRangeAnnotation,
    patrol_types: PatrolTypesAnnotation,
    status: PatrolStatusAnnotation = None,
    raise_on_empty: RaiseOnEmptyAnnotation = True,
    sub_page_size: SubPageSizeAnnotation = 100,
    patrols_overlap_daterange: PatrolsOverlapDateRangeAnnotation = True,
) -> PatrolsDF | EmptyDataFrame:
    if status is None:
        status = ["done"]

    patrols = client.get_patrols(
        since=time_range.since.isoformat(),
        until=time_range.until.isoformat(),
        patrol_type_value=patrol_types,
        status=status,
        sub_page_size=sub_page_size,
        patrols_overlap_daterange=patrols_overlap_daterange,
    )

    if raise_on_empty and patrols.empty:
        raise ValueError("No data returned from EarthRanger for get_patrols")

    return cast(PatrolsDF, patrols)


@register(tags=["io"])
def get_patrol_observations_from_patrols_df(
    client: EarthRangerClient,
    patrols_df: PatrolsDF,
    include_patrol_details: IncludePatrolDetailsAnnotation = True,
    raise_on_empty: RaiseOnEmptyAnnotation = True,
    sub_page_size: SubPageSizeAnnotation = 100,
) -> PatrolObservationsGDF | EmptyDataFrame:
    """Get observations for a patrol type from EarthRanger."""
    from ecoscope.relocations import Relocations

    if warehouse_client := _make_warehouse_client_from_env(
        er_site_url=client.server,
        er_api_token=SecretStr(client.token) if client.token else None,
    ):
        import geopandas as gpd  # type: ignore[import-untyped]

        table = warehouse_client.get_patrol_observations(
            patrols_df=patrols_df,
            include_patrol_details=include_patrol_details,
            sub_page_size=sub_page_size,
        )
        patrol_obs_relocs = gpd.GeoDataFrame.from_arrow(table)
    else:
        patrol_obs_relocs = client.get_patrol_observations(
            patrols_df=patrols_df,
            include_patrol_details=include_patrol_details,
            sub_page_size=sub_page_size,
        )
        if isinstance(patrol_obs_relocs, Relocations):
            patrol_obs_relocs = patrol_obs_relocs.gdf

    if raise_on_empty and patrol_obs_relocs.empty:
        raise ValueError("No data returned from EarthRanger for get_patrol_observations_with_patrol_filter")

    return cast(PatrolObservationsGDF, patrol_obs_relocs)


@register()
def unpack_events_from_patrols_df(
    patrols_df: PatrolsDF,
    event_types: EventTypesAnnotation,
    time_range: TimeRangeAnnotation,
    include_null_geometry: IncludeNullGeometryAnnotation = True,
    truncate_to_time_range: TruncateToTimeRangeAnnotation = True,
    raise_on_empty: RaiseOnEmptyAnnotation = True,
) -> EventGDF | EmptyDataFrame:
    from ecoscope.io.earthranger_utils import (
        unpack_events_from_patrols_df,
    )

    patrol_events = unpack_events_from_patrols_df(
        patrols_df=patrols_df,
        event_type=event_types,
        drop_null_geometry=not include_null_geometry,
    )

    if raise_on_empty and patrol_events.empty:
        raise ValueError("No event data in provided patrols_df")

    if truncate_to_time_range and not patrol_events.empty:
        patrol_events = patrol_events.loc[  # type: ignore[assignment]
            (patrol_events.time >= time_range.since) & (patrol_events.time <= time_range.until)
        ]

    return cast(EventGDF, patrol_events)


@register(tags=["io"])
def get_event_type_display_names_from_events(
    client: EarthRangerClient,
    events_gdf: EventGDF,
    append_category_names: AppendCategorySelectionAnnotation = "duplicates",
) -> EventsWithDisplayNamesGDF:
    events_gdf = client.get_event_type_display_names_from_events(  # type: ignore[assignment]
        events_gdf=events_gdf,
        append_category_names=append_category_names,
    )
    return cast(EventsWithDisplayNamesGDF, events_gdf)


@dataclass
class CombinedEventsAndDetailsParams:
    client: str
    time_range: TimeRangeAnnotation
    event_type: SingleEventTypeAnnotation
    event_columns: EventColumnsAnnotation
    analysis_field: AnalysisFieldAnnotation
    analysis_field_label: AnalysisFieldLabelAnnotation
    analysis_field_unit: AnalysisFieldUnitAnnotation
    category_field: CategoryFieldAnnotation = ""
    category_field_label: CategoryFieldLabelAnnotation = ""
    include_null_geometry: IncludeNullGeometryAnnotation = True
    raise_on_empty: RaiseOnEmptyAnnotation = True
    include_details: IncludeDetailsAnnotation = False
    include_updates: IncludeUpdatesAnnotation = False
    include_related_events: IncludeRelatedEventsAnnotation = False
    include_display_values: IncludeDisplayValuesAnnotation = False

    def get_events_params(self):
        return {
            "client": self.client,
            "time_range": self.time_range,
            "event_types": [self.event_type],
            "event_columns": self.event_columns,
            "include_null_geometry": self.include_null_geometry,
            "raise_on_empty": self.raise_on_empty,
            "include_details": self.include_details,
            "include_updates": self.include_updates,
            "include_related_events": self.include_related_events,
            "include_display_values": self.include_display_values,
        }


@register()
def set_event_details_params(
    client: str,
    time_range: TimeRangeAnnotation,
    event_type: SingleEventTypeAnnotation,
    analysis_field: AnalysisFieldAnnotation,
    analysis_field_label: AnalysisFieldLabelAnnotation,
    analysis_field_unit: AnalysisFieldUnitAnnotation,
    event_columns: EventColumnsAnnotation = DefaultEventColumns,
    category_field: CategoryFieldAnnotation = "",
    category_field_label: CategoryFieldLabelAnnotation = "",
    include_null_geometry: IncludeNullGeometryAnnotation = True,
    raise_on_empty: RaiseOnEmptyAnnotation = True,
    include_details: IncludeDetailsAnnotation = True,
    include_updates: IncludeUpdatesAnnotation = False,
    include_related_events: IncludeRelatedEventsAnnotation = False,
    include_display_values: IncludeDisplayValuesAnnotation = False,
) -> Annotated[
    CombinedEventsAndDetailsParams,
    Field(description="Passthrough selected events settings for use in downstream tasks"),
]:
    return CombinedEventsAndDetailsParams(
        client=client,
        time_range=time_range,
        event_type=event_type,
        event_columns=event_columns,
        analysis_field=analysis_field,
        analysis_field_label=analysis_field_label,
        analysis_field_unit=analysis_field_unit,
        category_field=category_field,
        category_field_label=category_field_label,
        include_null_geometry=include_null_geometry,
        raise_on_empty=raise_on_empty,
        include_details=include_details,
        include_updates=include_updates,
        include_related_events=include_related_events,
        include_display_values=include_display_values,
    )


@register(tags=["io"])
def get_events_from_combined_params(
    combined_params: CombinedEventsAndDetailsParams,
) -> EventGDF | EventsWithDisplayNamesGDF | EmptyDataFrame:
    return task(get_events).validate().call(**combined_params.get_events_params())


@register()
def get_analysis_field_from_event_details(
    combined_params: CombinedEventsAndDetailsParams,
) -> str:
    return combined_params.analysis_field


@register()
def get_analysis_field_label_from_event_details(
    combined_params: CombinedEventsAndDetailsParams,
) -> str:
    return combined_params.analysis_field_label


@register()
def get_analysis_field_unit_from_event_details(
    combined_params: CombinedEventsAndDetailsParams,
) -> str:
    return combined_params.analysis_field_unit


@register()
def get_category_field_from_event_details(
    combined_params: CombinedEventsAndDetailsParams,
) -> str | None:
    return combined_params.category_field


@register()
def get_category_field_label_from_event_details(
    combined_params: CombinedEventsAndDetailsParams,
) -> str | None:
    return combined_params.category_field_label


@register()
def get_event_type_from_event_details(
    combined_params: CombinedEventsAndDetailsParams,
) -> str | None:
    return combined_params.event_type


@register(tags=["io"])
def get_choices_from_v2_event_type(
    client: EarthRangerClient,
    event_type: SingleEventTypeAnnotation,
    choice_field: Annotated[
        str | SkipJsonSchema[None],
        Field(description="The choice field to lookup values from"),
    ],
) -> dict[str, str]:
    from ecoscope.io.earthranger import ERClientNotFound

    choices: dict[str, str] = {}
    try:
        choices = client.get_choices_from_v2_event_type(event_type, choice_field)
    except ERClientNotFound:
        pass

    return choices


@register(tags=["io"])
def get_spatial_features_group(
    client: EarthRangerClient,
    spatial_features_group_name: Annotated[str, Field(description="The name of the group to fetch")],
) -> RegionsGDF | EmptyDataFrame:
    spatial_features_group = client.get_spatial_features_group(
        spatial_features_group_name=spatial_features_group_name,
        spatial_features_group_id=None,
        with_group_data=True,
    )
    sfg = SpatialFeaturesGroup(**spatial_features_group)  # type: ignore[arg-type]
    regions_gdf = sfg.features
    regions_gdf["metadata"] = [{"id": sfg.id, "display_name": sfg.name}] * len(regions_gdf)  # type: ignore[assignment]
    return cast(RegionsGDF, regions_gdf)


@register(tags=["io"])
def get_fields_from_event_type_schema(
    client: EarthRangerClient,
    event_type: SingleEventTypeAnnotation,
) -> dict[str, str]:
    fields: dict[str, str] = client.get_fields_from_event_type_schema(event_type)
    return fields


@register(tags=["io"])
def download_grouped_event_attachments(
    client: EarthRangerClient,
    grouped_event_gdfs: Annotated[
        list[tuple[CompositeFilter | None, AnyDataFrame]],
        Field(
            description=(
                "A keyed iterable of (group key, dataframe) tuples as produced by "
                "`split_groups`. Each event's attachments are downloaded into a "
                "subfolder named `<key_hash>_<event_id>`, where `<key_hash>` is a "
                "short sha256 hash of the encoded group key — so the FE can match "
                "attachments to dashboard views."
            ),
        ),
        SkippedDependencyFallback(_fallback_to_empty_grouped),
    ],
    output_dir: Annotated[str, Field(description="Directory to save attachments.")],
    attachments_subdir: Annotated[
        str,
        AdvancedField(
            description="Subdirectory inside the output directory to store attachments.",
            default="attachments",
            title="Attachments Subdirectory",
        ),
    ] = "attachments",
    use_index_as_id: Annotated[
        bool,
        AdvancedField(
            description="Use the dataframe index as the event id when True.",
            default=True,
        ),
    ] = True,
    skip_download: Annotated[
        bool,
        Field(description="If True, skip downloading attachments."),
    ] = False,
) -> list[str]:
    """
    Download EarthRanger event attachments for grouped event dataframes.

    Group-aware task that fetches per-event files from EarthRanger. It takes a
    keyed iterable of `(group key, dataframe)` tuples — typically the output of
    `split_groups` — and writes each event's files to
    `output_dir/<attachments_subdir>/<key_hash>_<event_id>/<filename>`.

    `<key_hash>` is a short sha256 hash of the encoded group key, computed
    by the same helper used by `persist_grouped_dfs_for_results_download`,
    so the FE can match downloaded attachments to the dashboard view they
    belong to. Groups with a `None` key (e.g. the SkippedDependencyFallback
    sentinel) or an empty dataframe are skipped.

    Args:
        client: EarthRanger API client.
        grouped_event_gdfs: Keyed iterable of `(CompositeFilter, DataFrame)`
            tuples. Each DataFrame's index or `id` column supplies event IDs.
        output_dir: Base directory where attachments are written (supports `file://`).
        attachments_subdir: Folder name under `output_dir` for attachments.
        use_index_as_id: When True, use the DataFrame index as the event ID;
            otherwise use the `id` column.
        skip_download: If True, no requests are made and an empty list is returned.

    Returns:
        A flat list of relative paths (under `output_dir`) to every saved
        attachment across all groups.
    """
    from ecoscope.io.earthranger import ERClientNotFound  # type: ignore[import-untyped]

    if skip_download:
        return []
    attachments_root = f"{output_dir.rstrip('/')}/{attachments_subdir}"

    downloaded_paths: list[str] = []

    for composite_filter, event_gdf in grouped_event_gdfs:
        if composite_filter is None or event_gdf.empty:
            continue
        key_hash = _hash_grouper_key(composite_filter)

        for idx, row in tqdm(
            event_gdf.iterrows(),
            total=len(event_gdf),
            desc=f"Downloading attachments ({key_hash})",
        ):
            event_id = idx if use_index_as_id else row["id"]

            url = f"/activity/event/{event_id}/files"
            try:
                results = client._get(url)["results"]  # type: ignore[attr-defined]
            except ERClientNotFound:
                results = []

            if not results:
                continue

            event_dir_name = f"{key_hash}_{event_id}"
            event_dir = f"{attachments_root}/{event_dir_name}"

            for result in results:
                file_id = result["id"]
                filename = result["filename"]

                response = client._get(  # type: ignore[attr-defined]
                    f"activity/event/{event_id}/file/{file_id}/original/{filename}",
                    return_response=True,
                )

                _persist_bytes(response.content, event_dir, filename)

                downloaded_paths.append(f"{attachments_subdir}/{event_dir_name}/{filename}")

    return downloaded_paths


def _strip_v2_prefix(ref: str) -> str:
    path = ref.lstrip("/")
    if path.startswith("api/v2.0/"):
        return path[len("api/v2.0/") :]
    return ref


def _fetch_v2_schema(client: EarthRangerClient, event_type: str) -> dict:
    """Fetch a v2 event type schema and return in canonical format.

    The canonical format uses resolved properties with enumNames and an
    ordered_keys list derived from the property order.
    """

    def _fetch_enum_names(any_of: list) -> dict:
        enum_names = {}
        for definition in any_of:
            ref = definition.get("$ref")
            if not ref:
                continue
            try:
                # Strip v2 prefix to avoid double-prefixing in client._get
                ref_schema = client._get(_strip_v2_prefix(ref), params={"s_format": "oneOf"})  # type: ignore[attr-defined]
                for choice in ref_schema.get("oneOf", []):
                    const = choice.get("const")
                    title = choice.get("title")
                    if const is not None:
                        enum_names[const] = title or const
                # Fallback to parallel enum/enumNames lists
                if not enum_names:
                    for val, label in zip(ref_schema.get("enum", []), ref_schema.get("enumNames", [])):
                        enum_names[val] = label
            except Exception:
                continue
        return enum_names

    with client._use_v2_api():  # type: ignore[attr-defined]
        v2_schema = client._get(  # type: ignore[attr-defined]
            f"activity/eventtypes/{event_type}/schema"
        )
        json_schema = v2_schema.get("json", {})
        properties = json_schema.get("properties", {})

        # Collect properties from allOf conditional blocks.
        for condition in json_schema.get("allOf", []):
            for key, prop in condition.get("then", {}).get("properties", {}).items():
                if key not in properties:
                    properties[key] = dict(prop)

        # Resolve anyOf/$ref choice references into enumNames dicts
        for prop in properties.values():
            # Top-level anyOf — scalar enum fields
            any_of = prop.get("anyOf")
            if any_of:
                enum_names = _fetch_enum_names(any_of)
                if enum_names:
                    prop["enumNames"] = enum_names

            items_def = prop.get("items", {})

            # items.anyOf — simple array enum field
            items_any_of = items_def.get("anyOf")
            if items_any_of:
                enum_names = _fetch_enum_names(items_any_of)
                if enum_names:
                    items_def["enumNames"] = enum_names

            # items.properties — array-of-objects (sub-forms)
            for item_prop in items_def.get("properties", {}).values():
                item_any_of = item_prop.get("anyOf")
                if item_any_of:
                    enum_names = _fetch_enum_names(item_any_of)
                    if enum_names:
                        item_prop["enumNames"] = enum_names
                # Multi-select arrays within sub-form fields
                nested_items_def = item_prop.get("items", {})
                nested_any_of = nested_items_def.get("anyOf")
                if nested_any_of:
                    enum_names = _fetch_enum_names(nested_any_of)
                    if enum_names:
                        nested_items_def["enumNames"] = enum_names

    return {
        "properties": properties,
        "ordered_keys": list(properties.keys()),
    }


def _normalize_v1_schema(v1_schema: dict) -> dict:
    """Convert a v1 schema response to canonical v2 format.

    Merges definition titleMap entries into enumNames on the corresponding
    properties, and derives ordered_keys from the definition order.
    """
    properties = dict(v1_schema.get("schema", {}).get("properties", {}))
    definitions = v1_schema.get("definition", [])

    ordered_keys: list[str] = []

    def _merge_title_map(key: str, item: dict, props: dict) -> None:
        """Merge a definition item's titleMap/title into the target properties dict."""
        if key not in props:
            return
        title_map = item.get("titleMap")
        if title_map:
            enum_names = {
                entry["value"]: entry["name"]
                for entry in title_map
                if isinstance(entry, dict) and "value" in entry and "name" in entry
            }
            if enum_names:
                props[key] = {**props[key], "enumNames": enum_names}
        def_title = item.get("title")
        if def_title and "title" not in props[key]:
            props[key] = {**props[key], "title": def_title}

    def walk_definitions(items: list, props: dict) -> None:
        for item in items:
            if not isinstance(item, dict):
                continue
            key = item.get("key")
            if key:
                ordered_keys.append(key)
                _merge_title_map(key, item, props)
                # Recurse into nested items (sub-forms) with nested properties
                nested = item.get("items")
                if isinstance(nested, list) and key in props:
                    nested_props = props[key].get("items", {}).get("properties", {})
                    walk_definitions(nested, nested_props)
            else:
                # Section without key — recurse with same properties context
                nested = item.get("items")
                if isinstance(nested, list):
                    walk_definitions(nested, props)

    walk_definitions(definitions, properties)

    return {
        "properties": properties,
        "ordered_keys": ordered_keys,
    }


def _resolve_event_details(
    details: dict, schema: dict | None, map_to_titles: bool = True, ordered: bool = True
) -> dict:
    """Process event details using a canonical v2 schema.

    Resolves enum values via enumNames and optionally maps keys to titles.
    """
    if not details:
        return {}

    if not schema:
        return details

    properties = schema.get("properties", {})

    if ordered:
        ordered_keys = list(schema.get("ordered_keys", []))
        # Append keys from details that are not in the schema ordering
        for k in details.keys():
            if k not in ordered_keys and k != "updates":
                ordered_keys.append(k)
    else:
        ordered_keys = list(details.keys())

    def resolve(value, prop_def):
        enum_names = prop_def.get("enumNames") or prop_def.get("items", {}).get("enumNames")
        if isinstance(enum_names, dict) and enum_names:
            if isinstance(value, list):
                return [enum_names.get(v, v) for v in value]
            return enum_names.get(value, value)
        return value

    result = {}
    for key in ordered_keys:
        if key not in details:
            continue
        value = details[key]

        prop_def = properties.get(key, {})
        resolved_value = resolve(value, prop_def)

        # Handle nested lists of objects (sub-forms)
        if isinstance(resolved_value, list) and len(resolved_value) > 0 and isinstance(resolved_value[0], dict):
            item_props = prop_def.get("items", {}).get("properties", {})
            new_list = []
            for item in resolved_value:
                new_item = {}
                for ik, iv in item.items():
                    ip_def = item_props.get(ik, {})
                    ititle = ip_def.get("title", ik) if map_to_titles else ik
                    new_item[ititle] = resolve(iv, ip_def)
                new_list.append(new_item)
            resolved_value = new_list

        new_key = prop_def.get("title", key) if map_to_titles else key
        result[new_key] = resolved_value

    return result


@register(tags=["io"])
def process_events_details(
    df: AnyDataFrame,
    client: EarthRangerClient,
    map_to_titles: bool = True,
    ordered: bool = True,
) -> AnyDataFrame:
    """
    Extracts and processes event details for all events in the DataFrame,
    resolving enum values from the schema for each event type and mapping keys to titles.
    """
    if df.empty or "event_details" not in df.columns or "event_type" not in df.columns:
        return df

    from ecoscope.io.earthranger import ERClientNotFound

    # Cache schemas to avoid repeated API calls
    schemas: dict = {}

    def get_schema(event_type):
        if event_type not in schemas:
            try:
                # Try v2 endpoint first → returns canonical format
                schemas[event_type] = _fetch_v2_schema(client, event_type)
            except ERClientNotFound:
                # Fall back to v1 endpoint → normalize to canonical format
                try:
                    v1_schema = client._get(  # type: ignore[attr-defined]
                        f"activity/events/schema/eventtype/{event_type}"
                    )
                    schemas[event_type] = _normalize_v1_schema(v1_schema)
                except Exception:
                    schemas[event_type] = {}
            except Exception:
                schemas[event_type] = {}
        return schemas[event_type]

    def process_row(row):
        details = row.get("event_details")
        if not isinstance(details, dict):
            return details

        event_type = row.get("event_type")
        if not event_type:
            return details

        schema = get_schema(event_type)
        return _resolve_event_details(details, schema, map_to_titles=map_to_titles, ordered=ordered)

    # Use apply to process each row
    df["event_details"] = df.apply(process_row, axis=1)
    return df
