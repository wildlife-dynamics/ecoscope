import typing
import geopandas as gpd
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


def clean_kwargs(addl_kwargs=None, **kwargs):
    if addl_kwargs is None:
        addl_kwargs = {}

    for k in addl_kwargs.keys():
        print(f"Warning: {k} is a non-standard parameter. Results may be unexpected.")
    return {k: v for k, v in {**addl_kwargs, **kwargs}.items() if v is not None}


def normalize_column(df, col):
    print(col)
    for k, v in pd.json_normalize(df.pop(col), sep="__").add_prefix(f"{col}__").items():
        df[k] = v.values


def dataframe_to_dict(events):
    if isinstance(events, gpd.GeoDataFrame):
        events["location"] = pd.DataFrame({"longitude": events.geometry.x, "latitude": events.geometry.y}).to_dict(
            "records"
        )
        del events["geometry"]

    if isinstance(events, pd.DataFrame):
        events = events.to_dict("records")
    return events


def to_gdf(df):
    longitude, latitude = (0, 1) if isinstance(df["location"].iat[0], list) else ("longitude", "latitude")
    return gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df["location"].str[longitude], df["location"].str[latitude]),
        crs=4326,
    )


def clean_time_cols(df):
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


def to_hex(val, default="#ff0000"):
    if val and not pd.isnull(val):
        return "#{:02X}{:02X}{:02X}".format(*[int(i) for i in val.split(",")])
    return default


def pack_columns(dataframe: pd.DataFrame, columns: typing.List):
    """This method would add all extra columns to single column"""
    metadata_cols = list(set(dataframe.columns).difference(set(columns)))

    # To prevent additional column from being dropped, name the column metadata (rename it back).
    if metadata_cols:
        dataframe["metadata"] = dataframe[metadata_cols].to_dict(orient="records")
        dataframe.drop(metadata_cols, inplace=True, axis=1)
        dataframe.rename(columns={"metadata": "additional"}, inplace=True)
    return dataframe


def geometry_from_event_geojson(
    df: pd.DataFrame, geojson_column="geojson", force_point_geometry=True, drop_null_geometry=True
):
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
