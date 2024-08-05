import geopandas as gpd
import pandas as pd
from dateutil import parser


def clean_kwargs(addl_kwargs={}, **kwargs):
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
    time_cols = ["time", "created_at", "updated_at", "end_time", "last_position_date", "recorded_at"]
    for col in time_cols:
        if col in df.columns:
            # convert x is not None to pd.isna(x) is False
            df[col] = df[col].apply(lambda x: pd.to_datetime(parser.parse(x)) if not pd.isna(x) else None)
    return df
