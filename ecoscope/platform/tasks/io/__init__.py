"""Data loading and persistence tasks for external services.

Provides functions for fetching data from EarthRanger, Google Earth Engine, and
SMART, as well as downloading raster regions-of-interest and persisting
DataFrames or text to local or cloud storage.
"""

from ._downloader import download_roi
from ._earthengine import calculate_ndvi_range, determine_season_windows
from ._earthranger import (
    get_analysis_field_from_event_details,
    get_analysis_field_label_from_event_details,
    get_analysis_field_unit_from_event_details,
    get_category_field_from_event_details,
    get_category_field_label_from_event_details,
    get_choices_from_v2_event_type,
    get_event_type_display_names_from_events,
    get_event_type_from_event_details,
    get_events,
    get_events_from_combined_params,
    get_fields_from_event_type_schema,
    get_patrol_events,
    get_patrol_events_from_combined_params,
    get_patrol_observations,
    get_patrol_observations_from_combined_params,
    get_patrol_observations_from_patrols_df,
    get_patrol_observations_from_patrols_df_and_combined_params,
    get_patrols,
    get_patrols_from_combined_params,
    get_spatial_features_group,
    get_subjectgroup_observations,
    set_event_details_params,
    set_patrol_status,
    set_patrol_types,
    set_patrols_and_patrol_events_params,
    unpack_events_from_patrols_df,
    unpack_events_from_patrols_df_and_combined_params,
)
from ._persist import persist_df, persist_text
from ._set_connection import set_er_connection, set_gee_connection, set_smart_connection
from ._smart import get_events_from_smart, get_patrol_observations_from_smart

__all__ = [
    "download_roi",
    "calculate_ndvi_range",
    "determine_season_windows",
    "get_analysis_field_from_event_details",
    "get_analysis_field_label_from_event_details",
    "get_analysis_field_unit_from_event_details",
    "get_category_field_from_event_details",
    "get_category_field_label_from_event_details",
    "get_choices_from_v2_event_type",
    "get_event_type_display_names_from_events",
    "get_event_type_from_event_details",
    "get_events",
    "get_events_from_combined_params",
    "get_fields_from_event_type_schema",
    "get_patrol_events",
    "get_patrol_events_from_combined_params",
    "get_patrol_observations",
    "get_patrol_observations_from_combined_params",
    "get_patrol_observations_from_patrols_df",
    "get_patrol_observations_from_patrols_df_and_combined_params",
    "get_patrols",
    "get_patrols_from_combined_params",
    "get_spatial_features_group",
    "get_subjectgroup_observations",
    "set_event_details_params",
    "set_patrol_status",
    "set_patrol_types",
    "set_patrols_and_patrol_events_params",
    "unpack_events_from_patrols_df",
    "unpack_events_from_patrols_df_and_combined_params",
    "persist_df",
    "persist_text",
    "set_er_connection",
    "set_gee_connection",
    "set_smart_connection",
    "get_events_from_smart",
    "get_patrol_observations_from_smart",
]
