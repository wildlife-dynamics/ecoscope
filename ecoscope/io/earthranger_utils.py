import geopandas as gpd  # type: ignore[import-untyped]
import pandas as pd
from dateutil import parser
from shapely.geometry import shape

TIME_COLS = [
    "time",
    "created_at",
    "updated_at",
    "end_time",
    "last_position_date",
    "recorded_at",
    "fixtime",
]


def clean_kwargs(addl_kwargs: dict | None = None, **kwargs) -> dict:
    if addl_kwargs is None:
        addl_kwargs = {}

    for k in addl_kwargs.keys():
        print(f"Warning: {k} is a non-standard parameter. Results may be unexpected.")
    return {k: v for k, v in {**addl_kwargs, **kwargs}.items() if v is not None}


def normalize_column(df: pd.DataFrame, col: str) -> None:
    normalized = pd.json_normalize(df.pop(col).to_list(), sep="__").add_prefix(f"{col}__")

    new_cols = sorted(normalized.columns.tolist())
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


def clean_time_cols(df: pd.DataFrame | gpd.GeoDataFrame) -> pd.DataFrame | gpd.GeoDataFrame:
    for col in TIME_COLS:
        if col in df.columns and not pd.api.types.is_datetime64_ns_dtype(df[col]):
            # convert x is not None to pd.isna(x) is False
            df[col] = df[col].apply(lambda x: pd.to_datetime(parser.parse(x), utc=True) if not pd.isna(x) else None)
    return df


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
