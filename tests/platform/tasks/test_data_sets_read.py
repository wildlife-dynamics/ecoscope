import json
from pathlib import Path

import geopandas as gpd  # type: ignore[import-untyped]
import pytest

from ecoscope.platform.data_sets import DataSetSchemaError
from ecoscope.platform.tasks.data_sets import gather_data_sets, read_data_set_manifest, read_data_set_table


@pytest.fixture
def data_set_root(tmp_path, patrol_events_gdf, patrol_legs_df):
    gather_data_sets(
        data_set="patrols",
        tables={"patrol_events": patrol_events_gdf, "patrol_legs": patrol_legs_df},
        root_path=str(tmp_path),
    )
    return str(tmp_path)


@pytest.fixture
def roots(data_set_root):
    return json.dumps({"patrol_events": data_set_root, "patrol_legs": data_set_root})


def test_read_table_returns_geodataframe_for_spatial_type(roots, patrol_events_gdf):
    gdf = read_data_set_table(data_set_roots=roots, table_type="patrol_events")
    assert isinstance(gdf, gpd.GeoDataFrame) and len(gdf) == len(patrol_events_gdf)


def test_read_table_accepts_file_url_root(data_set_root):
    roots = json.dumps({"patrol_legs": Path(data_set_root).as_uri()})
    assert len(read_data_set_table(data_set_roots=roots, table_type="patrol_legs")) == 2


def test_read_table_type_absent_from_roots_lists_keys(roots):
    with pytest.raises(DataSetSchemaError, match=r"patrol_tracks.*bound: patrol_events, patrol_legs"):
        read_data_set_table(data_set_roots=roots, table_type="patrol_tracks")


def test_read_table_missing_from_manifest_names_available_types(data_set_root):
    roots = json.dumps({"patrol_tracks": data_set_root})
    with pytest.raises(DataSetSchemaError, match=r"patrol_tracks.*available: patrol_events, patrol_legs"):
        read_data_set_table(data_set_roots=roots, table_type="patrol_tracks")


def test_read_table_missing_manifest_names_type_and_root(tmp_path):
    root = str(tmp_path / "empty")
    with pytest.raises(DataSetSchemaError, match=r"patrol_legs.*no data set manifest.*empty"):
        read_data_set_table(data_set_roots=json.dumps({"patrol_legs": root}), table_type="patrol_legs")


def test_optional_read_returns_empty_frame_when_unbound(roots):
    gdf = read_data_set_table(data_set_roots=roots, table_type="spatial_feature_groups", optional=True)
    assert isinstance(gdf, gpd.GeoDataFrame) and len(gdf) == 0 and "pk" in gdf.columns


def test_optional_read_returns_plain_dataframe_for_non_spatial_type():
    df = read_data_set_table(data_set_roots="{}", table_type="patrol_legs", optional=True)
    assert not isinstance(df, gpd.GeoDataFrame) and len(df) == 0 and "patrol_id" in df.columns


def test_optional_read_still_raises_for_bound_root_without_manifest(tmp_path):
    roots = json.dumps({"patrol_legs": str(tmp_path / "nope")})
    with pytest.raises(DataSetSchemaError, match="no data set manifest"):
        read_data_set_table(data_set_roots=roots, table_type="patrol_legs", optional=True)


def test_read_table_rejects_non_object_roots():
    with pytest.raises(DataSetSchemaError, match="not a JSON object"):
        read_data_set_table(data_set_roots="[]", table_type="patrol_tracks")


def test_read_table_rejects_invalid_json_roots():
    with pytest.raises(DataSetSchemaError, match="not a JSON object"):
        read_data_set_table(data_set_roots="{nope", table_type="patrol_tracks")


def test_read_table_rejects_newer_schema_version(data_set_root):
    p = Path(data_set_root) / "data_sets" / "manifest.json"
    m = json.loads(p.read_text())
    m["tables"]["patrol_legs"]["schema_version"] = "2"
    p.write_text(json.dumps(m))
    with pytest.raises(DataSetSchemaError, match=r"schema_version.*expected 1.*found 2"):
        read_data_set_table(data_set_roots=json.dumps({"patrol_legs": data_set_root}), table_type="patrol_legs")


def test_read_table_returns_empty_frame_for_zero_rows(tmp_path, patrol_events_gdf):
    gather_data_sets(data_set="p", tables={"patrol_events": patrol_events_gdf.iloc[0:0]}, root_path=str(tmp_path))
    roots = json.dumps({"patrol_events": str(tmp_path)})
    assert len(read_data_set_table(data_set_roots=roots, table_type="patrol_events")) == 0


def test_manifest_lists_tables(data_set_root):
    manifest = read_data_set_manifest(data_set_root=data_set_root)
    assert manifest.data_set == "patrols" and set(manifest.tables) == {"patrol_events", "patrol_legs"}
