from typing import Any

import geopandas as gpd  # type: ignore[import-untyped]
import pandas as pd
from shapely.geometry import shape

from ecoscope.io.utils import clean_time_cols


def clean_kwargs(addl_kwargs: dict | None = None, **kwargs) -> dict:
    if addl_kwargs is None:
        addl_kwargs = {}

    for k in addl_kwargs.keys():
        print(f"Warning: {k} is a non-standard parameter. Results may be unexpected.")
    return {k: v for k, v in {**addl_kwargs, **kwargs}.items() if v is not None}


def normalize_column(df: pd.DataFrame, col: str, sort_columns: bool = False) -> None:
    normalized = pd.json_normalize(df.pop(col).to_list(), sep="__").add_prefix(f"{col}__")

    new_cols = normalized.columns.tolist()
    if sort_columns:
        new_cols = sorted(new_cols)

    for k in new_cols:
        df[k] = normalized[k].values  # type: ignore[call-overload]


def dataframe_to_dict_or_list(events: gpd.GeoDataFrame | pd.DataFrame | dict | list[dict]) -> dict | list[dict]:
    if isinstance(events, gpd.GeoDataFrame):
        events["location"] = pd.DataFrame({"longitude": events.geometry.x, "latitude": events.geometry.y}).to_dict(
            "records"
        )
        del events["geometry"]

    if isinstance(events, pd.DataFrame) or isinstance(events, gpd.GeoDataFrame):
        processed_events = events.to_dict("records")
    else:
        processed_events = events
    return processed_events


def to_gdf(df: pd.DataFrame) -> gpd.GeoDataFrame:
    longitude, latitude = (0, 1) if isinstance(df["location"].iat[0], list) else ("longitude", "latitude")
    return gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df["location"].str[longitude], df["location"].str[latitude]),  # type: ignore[index]
        crs=4326,
    )


def format_iso_time(date_string: str) -> str:
    try:
        return pd.to_datetime(date_string).isoformat()
    except ValueError:
        raise ValueError(f"Failed to parse timestamp'{date_string}'")


def to_hex(val: str | None, default: str = "#ff0000") -> str:
    if val and not pd.isnull(val):
        return "#{:02X}{:02X}{:02X}".format(*[int(i) for i in val.split(",")])
    return default


def pack_columns(dataframe: pd.DataFrame, columns: list) -> pd.DataFrame:
    """This method would add all extra columns to single column"""
    metadata_cols = list(set(dataframe.columns).difference(set(columns)))

    # To prevent additional column from being dropped, name the column metadata (rename it back).
    if metadata_cols:
        dataframe["metadata"] = dataframe[metadata_cols].to_dict(orient="records")  # type: ignore[assignment]
        dataframe.drop(metadata_cols, inplace=True, axis=1)
        if "additional" in dataframe.columns:
            result = []
            for _, row in dataframe.iterrows():
                add_dict = row["additional"]
                meta_dict = row["metadata"]
                merged = {**add_dict, **meta_dict}
                result.append(merged)
            dataframe["additional"] = result  # type: ignore[assignment]
        else:
            dataframe.rename(columns={"metadata": "additional"}, inplace=True)
    return dataframe


def geometry_from_event_geojson(
    df: pd.DataFrame,
    geojson_column: str = "geojson",
    force_point_geometry: bool = True,
    drop_null_geometry: bool = True,
) -> gpd.GeoDataFrame:
    if df.empty:
        return gpd.GeoDataFrame()

    def shape_from_geojson(geojson):
        try:
            result = shape(geojson.get("geometry"))
        except Exception:
            return None

        return result.centroid if force_point_geometry else result

    df["geometry"] = df[geojson_column].apply(shape_from_geojson)
    if drop_null_geometry:
        df = df.dropna(subset="geometry").reset_index()

    return df


def unpack_events_from_patrols_df(
    patrols_df: pd.DataFrame,
    event_type: list[str] | None = None,
    force_point_geometry: bool = True,
    drop_null_geometry: bool = True,
) -> gpd.GeoDataFrame:
    events = []
    for _, row in patrols_df.iterrows():
        for segment in row.get("patrol_segments", []):
            for event in segment.get("events", []):
                if event_type is None or event_type == [] or event.get("event_type") in event_type:
                    event["patrol_id"] = row.get("id")
                    event["patrol_serial_number"] = row.get("serial_number")
                    event["patrol_segment_id"] = segment.get("id")
                    event["patrol_start_time"] = (segment.get("time_range") or {}).get("start_time")
                    event["patrol_type"] = segment.get("patrol_type")
                    event["patrol_subject"] = (segment.get("leader") or {}).get("name")
                    events.append(event)
    events_df = pd.DataFrame(events)
    if events_df.empty:
        return events_df

    events_df = geometry_from_event_geojson(
        events_df, force_point_geometry=force_point_geometry, drop_null_geometry=drop_null_geometry
    )
    events_df["time"] = events_df["geojson"].apply(
        lambda x: x.get("properties", {}).get("datetime") if isinstance(x, dict) else None
    )
    events_df = events_df.dropna(subset="time").reset_index()
    events_df = clean_time_cols(events_df)

    return gpd.GeoDataFrame(events_df, geometry="geometry", crs=4326)


def _synthesize_event_geojson(event: dict) -> dict:
    """Build an ER-native ``geojson`` Feature for a nested warehouse patrol event.

    Resurrects the logic removed from the io-core client (PR #24): builds
    ``{"type": "Feature", "geometry": ..., "properties": {"datetime": ...}}`` from
    the event's WKB ``geometry`` and ``event_time`` so ecoscope's
    ``unpack_events_from_patrols_df`` (which reads ``event["geojson"]`` and
    ``geojson["properties"]["datetime"]``) can consume it unchanged.

    ``geometry`` is ``None`` when the event has none; ``datetime`` is a tz-aware
    ISO string (or ``None``). ``event_time`` may be a (tz-aware) datetime/Timestamp
    or raw int64 nanoseconds-since-epoch (UTC by schema contract) -- both handled.
    """
    import shapely.geometry
    import shapely.wkb

    geometry_wkb = event.get("geometry")
    if geometry_wkb is not None:
        geometry = shapely.geometry.mapping(shapely.wkb.loads(geometry_wkb))
    else:
        geometry = None

    event_time = event.get("event_time")
    if event_time is None or pd.isna(event_time):
        datetime_str = None
    elif hasattr(event_time, "isoformat"):
        datetime_str = event_time.isoformat()
    else:
        datetime_str = pd.Timestamp(event_time, unit="ns", tz="UTC").isoformat()

    return {
        "type": "Feature",
        "geometry": geometry,
        "properties": {"datetime": datetime_str},
    }


def warehouse_patrols_table_to_patrols_df(table: Any) -> pd.DataFrame:
    """Convert a warehouse nested-patrols ``pa.Table`` into the ER-native PatrolsDF.

    Takes the ``PATROLS_WITH_EVENTS_NESTED_SCHEMA_V1`` table returned by
    ``ERWarehouseClient.get_patrols`` (events nested under
    ``patrol_segments[].events[]``, event geometry = WKB) and reshapes it into the
    dict shape that ecoscope's ``unpack_events_from_patrols_df`` and
    ``get_patrol_observations_from_patrols_df`` already consume, so both downstream
    consumers work unchanged:

    - each event gets a synthesized ``geojson`` (see ``_synthesize_event_geojson``);
    - each segment gets ``time_range = {"start_time": <time_range_start>,
      "end_time": <time_range_end>}`` (the warehouse serves flat
      ``time_range_start``/``time_range_end`` fields);
    - ``patrol_type`` is already a flat segment field and is left as-is;
    - the warehouse serves the leader's ``leader_id`` and resolved ``leader_name``, so
      ``leader = {"id": <leader_id>, "name": <leader_name>}`` and ``patrol_subject``
      populates in ``unpack_events_from_patrols_df`` (``name`` is ``None`` when the
      leader is absent or unresolved).
    """
    records = table.to_pylist()
    for patrol in records:
        for segment in patrol.get("patrol_segments") or []:
            segment["time_range"] = {
                "start_time": segment.get("time_range_start"),
                "end_time": segment.get("time_range_end"),
            }
            segment["leader"] = {"id": segment.get("leader_id"), "name": segment.get("leader_name")}
            for event in segment.get("events") or []:
                event["geojson"] = _synthesize_event_geojson(event)
    return pd.DataFrame(records)


def append_event_type_display_names(
    events_df: pd.DataFrame,
    event_types_table: Any,
    *,
    append_category_names: str = "duplicates",
) -> pd.DataFrame:
    """Append an ``event_type_display`` column using a warehouse event-types table.

    Mirrors ``EarthRangerIO.get_event_type_display_names_from_events`` but resolves
    display names from a ``get_event_types()`` ``pa.Table`` (``EVENT_TYPES_SCHEMA_V1``:
    ``value``, ``display``, ``category_value``, ``category_display``) instead of the
    legacy API.

    Builds ``value -> display`` and ``value -> category`` maps, appends
    ``event_type_display``, and applies ``append_category_names`` semantics:

    - ``"never"``: never append a category.
    - ``"always"``: always append a category.
    - ``"duplicates"`` (default): append only for event types whose display names
      collide.

    The category name uses the warehouse ``category_display``, falling back to the
    category slug (``category_value``) when the display is null or an older API omits
    the column. Event types missing from the registry (orphans) fall back to their raw
    ``event_type`` value as the display.
    """
    assert "event_type" in events_df.columns

    event_types = (
        event_types_table.to_pandas() if hasattr(event_types_table, "to_pandas") else pd.DataFrame(event_types_table)
    )
    display_lookup = dict(zip(event_types["value"], event_types["display"]))
    events_df["event_type_display"] = events_df["event_type"].map(display_lookup).fillna(events_df["event_type"])

    has_duplicates = len(events_df["event_type_display"].unique()) != len(events_df["event_type"].unique())
    do_append = append_category_names == "always" or (append_category_names == "duplicates" and has_duplicates)
    if not do_append:
        return events_df

    # Prefer the warehouse category_display; fall back to the category slug when the
    # display is null (or an older API omits the column).
    # Orphan event types (absent from the registry) have no category, so fall
    # back to the raw event_type value rather than letting a NaN propagate through
    # the string concat and null the whole event_type_display (which StrictEventsGDFSchema rejects).
    if "category_display" in event_types.columns:
        category = event_types["category_display"].fillna(event_types["category_value"])
    else:
        category = event_types["category_value"]
    category_lookup = dict(zip(event_types["value"], category))
    if append_category_names == "duplicates":
        is_duplicate_display = events_df.groupby("event_type_display")["event_type"].transform("nunique") > 1
        dup_event_types = events_df.loc[is_duplicate_display, "event_type"]
        category_display = dup_event_types.map(category_lookup).fillna(dup_event_types)
        events_df.loc[is_duplicate_display, "event_type_display"] = (
            events_df.loc[is_duplicate_display, "event_type_display"] + " (" + category_display + ")"
        )
    else:
        category_display = events_df["event_type"].map(category_lookup).fillna(events_df["event_type"])
        events_df["event_type_display"] = events_df["event_type_display"] + " (" + category_display + ")"

    return events_df
