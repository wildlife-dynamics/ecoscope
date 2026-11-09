import geopandas as gpd
import pandas as pd
import pytest

from ecoscope.platform.data_sets import (
    DataSetSchemaError,
    frame_to_table,
    read_table,
    table_to_frame,
    write_table,
)


def test_geodataframe_round_trips_with_tz_aware_timestamps(tmp_path, patrol_events_gdf):
    table = frame_to_table(patrol_events_gdf, "patrol_events")
    read_path, rows = write_table(table, str(tmp_path), "patrol_events")
    assert rows == len(patrol_events_gdf) and read_path.endswith("data_sets/patrol_events.arrow")
    back = table_to_frame(read_table(str(tmp_path), "patrol_events", "1"), "patrol_events")
    assert isinstance(back, gpd.GeoDataFrame)
    assert str(back["time"].dt.tz) == str(patrol_events_gdf["time"].dt.tz)
    assert back.geometry.equals(patrol_events_gdf.geometry.reset_index(drop=True))


def test_all_null_geometry_round_trips(tmp_path, patrol_events_gdf):
    gdf = patrol_events_gdf.copy()
    gdf["geometry"] = None
    write_table(frame_to_table(gdf, "patrol_events"), str(tmp_path), "patrol_events")
    back = table_to_frame(read_table(str(tmp_path), "patrol_events", "1"), "patrol_events")
    assert back.geometry.isna().all()


def test_plain_dataframe_table_type_returns_dataframe(tmp_path, patrol_legs_df):
    write_table(frame_to_table(patrol_legs_df, "patrol_legs"), str(tmp_path), "patrol_legs")
    back = table_to_frame(read_table(str(tmp_path), "patrol_legs", "1"), "patrol_legs")
    assert type(back) is pd.DataFrame and list(back.columns) == list(patrol_legs_df.columns)


def test_frame_to_table_rejects_wrong_shape(patrol_legs_df):
    with pytest.raises(DataSetSchemaError, match="patrol_events"):
        frame_to_table(patrol_legs_df, "patrol_events")
