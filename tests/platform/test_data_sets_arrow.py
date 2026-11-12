import geopandas as gpd
import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
from shapely.geometry import Point

from ecoscope.platform.codecs import TREE_COLUMNS_KEY, decode_dataframe, encode_dataframe
from ecoscope.platform.data_sets import (
    DataSetSchemaError,
    frame_to_table,
    read_table,
    table_filename,
    table_to_frame,
    write_table,
)


def test_geodataframe_round_trips_with_tz_aware_timestamps(tmp_path, patrol_events_gdf):
    table = frame_to_table(patrol_events_gdf, "patrol_events")
    read_path, rows = write_table(table, str(tmp_path), "patrol_events")
    assert rows == len(patrol_events_gdf) and read_path.endswith("data_sets/patrol_events.arrow")
    back = table_to_frame(
        read_table(str(tmp_path), "patrol_events", "1", table_filename("patrol_events")), "patrol_events"
    )
    assert isinstance(back, gpd.GeoDataFrame)
    assert str(back["time"].dt.tz) == str(patrol_events_gdf["time"].dt.tz)
    assert back.geometry.equals(patrol_events_gdf.geometry.reset_index(drop=True))


def test_all_null_geometry_round_trips(tmp_path, patrol_events_gdf):
    gdf = patrol_events_gdf.copy()
    gdf["geometry"] = None
    write_table(frame_to_table(gdf, "patrol_events"), str(tmp_path), "patrol_events")
    raw = read_table(str(tmp_path), "patrol_events", "1", table_filename("patrol_events"))
    assert raw.schema.field("geometry").type.extension_name == "geoarrow.wkb"
    back = table_to_frame(raw, "patrol_events")
    assert back.geometry.isna().all()
    assert back.crs == "EPSG:4326"


def test_plain_dataframe_table_type_returns_dataframe(tmp_path, patrol_legs_df):
    write_table(frame_to_table(patrol_legs_df, "patrol_legs"), str(tmp_path), "patrol_legs")
    back = table_to_frame(read_table(str(tmp_path), "patrol_legs", "1", table_filename("patrol_legs")), "patrol_legs")
    assert type(back) is pd.DataFrame and list(back.columns) == list(patrol_legs_df.columns)


def test_frame_to_table_rejects_wrong_shape(patrol_legs_df):
    with pytest.raises(DataSetSchemaError, match="patrol_events"):
        frame_to_table(patrol_legs_df, "patrol_events")


def test_nested_extra_column_round_trips_as_written(tmp_path, patrol_events_gdf):
    gdf = patrol_events_gdf.copy()
    details = [{"count": 1, "tags": ["a", "b"]}, {"count": "3"}, None]
    gdf["details"] = details
    table = frame_to_table(gdf, "patrol_events")
    assert table.schema.field("details").type == "string"
    write_table(table, str(tmp_path), "patrol_events")
    back = table_to_frame(
        read_table(str(tmp_path), "patrol_events", "1", table_filename("patrol_events")), "patrol_events"
    )
    assert back["details"].tolist() == details
    # What the step cache refused when Arrow handed nested cells back as arrays.
    assert decode_dataframe(encode_dataframe(back))["details"].tolist() == details


def test_numpy_arrays_inside_a_nested_cell_are_stored_as_lists(tmp_path, patrol_events_gdf):
    """GeoJSON read back from Parquet holds its coordinates as arrays."""
    gdf = patrol_events_gdf.copy()
    gdf["geojson"] = [
        {"type": "Point", "coordinates": np.array([36.8, -1.3]), "n": np.int64(2)},
        {"type": "Point", "coordinates": np.array([36.9, -1.4]), "n": np.int64(3)},
        None,
    ]
    write_table(frame_to_table(gdf, "patrol_events"), str(tmp_path), "patrol_events")
    back = table_to_frame(
        read_table(str(tmp_path), "patrol_events", "1", table_filename("patrol_events")), "patrol_events"
    )
    assert back["geojson"].tolist() == [
        {"type": "Point", "coordinates": [36.8, -1.3], "n": 2},
        {"type": "Point", "coordinates": [36.9, -1.4], "n": 3},
        None,
    ]
    # Lists, so the step cache takes the frame.
    assert decode_dataframe(encode_dataframe(back))["geojson"].tolist() == back["geojson"].tolist()


def test_registered_struct_column_stays_a_struct():
    gdf = gpd.GeoDataFrame(
        {
            "pk": ["a"],
            "name": ["North"],
            "short_name": ["N"],
            "feature_type": ["sector"],
            "metadata": [{"id": "g1", "display_name": "Sectors"}],
        },
        geometry=[Point(36.8, -1.3)],
        crs="EPSG:4326",
    )
    table = frame_to_table(gdf, "spatial_feature_groups")
    assert pa.types.is_struct(table.schema.field("metadata").type)
    assert TREE_COLUMNS_KEY not in (table.schema.metadata or {})


def test_frame_to_table_wraps_a_cell_json_cannot_carry(patrol_events_gdf):
    gdf = patrol_events_gdf.copy()
    gdf["details"] = [{"when": pd.Timestamp("2025-01-01")}, {}, {}]
    with pytest.raises(DataSetSchemaError, match=r"patrol_events.*details"):
        frame_to_table(gdf, "patrol_events")


def test_read_table_reads_the_path_it_is_given(tmp_path, patrol_legs_df):
    write_table(frame_to_table(patrol_legs_df, "patrol_legs"), str(tmp_path), "patrol_legs")
    (tmp_path / "data_sets" / "patrol_legs.arrow").rename(tmp_path / "legs.arrow")
    assert read_table(str(tmp_path), "patrol_legs", "1", "legs.arrow").num_rows == len(patrol_legs_df)


@pytest.mark.parametrize("path", ["/etc/passwd", "../other/patrol_legs.arrow"])
def test_read_table_rejects_a_path_outside_the_root(tmp_path, path):
    with pytest.raises(DataSetSchemaError, match="leaves the data set root"):
        read_table(str(tmp_path), "patrol_legs", "1", path)


def test_frame_to_table_wraps_plain_dataframe_for_spatial_type(patrol_events_gdf):
    with pytest.raises(DataSetSchemaError, match=r"patrol_events.*geometry"):
        frame_to_table(pd.DataFrame(patrol_events_gdf), "patrol_events")
