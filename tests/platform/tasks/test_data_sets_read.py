import json
from pathlib import Path

import geopandas as gpd  # type: ignore[import-untyped]
import pytest

from ecoscope.platform.data_sets import DataSetSchemaError
from ecoscope.platform.tasks.data_sets import gather_data_sets, read_data_set_manifest, read_data_set_table


def typed(tables):
    """Each frame as a table named after its type."""
    return {k: {"table_type": k, "frame": v} for k, v in tables.items()}


@pytest.fixture
def data_set_root(tmp_path, patrol_events_gdf, patrol_legs_df):
    gather_data_sets(
        data_set="patrols",
        tables=typed({"patrol_events": patrol_events_gdf, "patrol_legs": patrol_legs_df}),
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


def test_read_table_wraps_an_invalid_manifest(data_set_root):
    (Path(data_set_root) / "data_sets" / "manifest.json").write_text('{"data_set": "patrols"}')
    with pytest.raises(
        DataSetSchemaError, match=r"patrol_legs.*invalid data set manifest.*" + Path(data_set_root).name
    ):
        read_data_set_table(data_set_roots=json.dumps({"patrol_legs": data_set_root}), table_type="patrol_legs")


def test_read_table_follows_the_manifest_path(data_set_root):
    root = Path(data_set_root)
    (root / "data_sets" / "patrol_legs.arrow").rename(root / "data_sets" / "legs-v1.arrow")
    p = root / "data_sets" / "manifest.json"
    m = json.loads(p.read_text())
    m["tables"]["patrol_legs"]["path"] = "data_sets/legs-v1.arrow"
    p.write_text(json.dumps(m))
    df = read_data_set_table(data_set_roots=json.dumps({"patrol_legs": data_set_root}), table_type="patrol_legs")
    assert len(df) > 0


def test_read_table_wraps_a_missing_table_file(data_set_root):
    (Path(data_set_root) / "data_sets" / "patrol_legs.arrow").unlink()
    with pytest.raises(DataSetSchemaError, match=r"patrol_legs.*no table file.*" + Path(data_set_root).name):
        read_data_set_table(data_set_roots=json.dumps({"patrol_legs": data_set_root}), table_type="patrol_legs")


def test_read_table_returns_empty_frame_for_zero_rows(tmp_path, patrol_events_gdf):
    gather_data_sets(
        data_set="p", tables=typed({"patrol_events": patrol_events_gdf.iloc[0:0]}), root_path=str(tmp_path)
    )
    roots = json.dumps({"patrol_events": str(tmp_path)})
    assert len(read_data_set_table(data_set_roots=roots, table_type="patrol_events")) == 0


def test_manifest_lists_tables(data_set_root):
    manifest = read_data_set_manifest(data_set_root=data_set_root)
    assert manifest.data_set == "patrols" and set(manifest.tables) == {"patrol_events", "patrol_legs"}


@pytest.fixture
def two_legs_root(tmp_path, patrol_legs_df):
    gather_data_sets(
        data_set="patrols",
        tables={
            "legs_rangers": {"table_type": "patrol_legs", "frame": patrol_legs_df},
            "legs_vehicles": {"table_type": "patrol_legs", "frame": patrol_legs_df.iloc[:1]},
        },
        root_path=str(tmp_path),
    )
    return str(tmp_path)


def test_a_table_is_bound_by_type_whatever_it_is_called(two_legs_root, patrol_legs_df):
    """The workflow asks for a `patrol_legs` table; which one it gets is in the map."""
    roots = json.dumps({"patrol_legs": {"url": two_legs_root, "table_id": "legs_vehicles"}})
    assert len(read_data_set_table(data_set_roots=roots, table_type="patrol_legs")) == 1
    roots = json.dumps({"patrol_legs": {"url": two_legs_root, "table_id": "legs_rangers"}})
    assert len(read_data_set_table(data_set_roots=roots, table_type="patrol_legs")) == len(patrol_legs_df)


def test_a_workflow_may_read_two_tables_of_one_type(two_legs_root):
    roots = json.dumps({"legs_rangers": two_legs_root, "legs_vehicles": two_legs_root})
    read = {
        i: len(read_data_set_table(data_set_roots=roots, table_type="patrol_legs", binding=i))
        for i in ("legs_rangers", "legs_vehicles")
    }
    assert read["legs_vehicles"] == 1 and read["legs_rangers"] > 1


def test_read_refuses_a_bound_table_of_another_type(data_set_root):
    roots = json.dumps({"patrol_legs": {"url": data_set_root, "table_id": "patrol_events"}})
    with pytest.raises(DataSetSchemaError, match=r"expected a patrol_legs table.*patrol_events.*is a patrol_events"):
        read_data_set_table(data_set_roots=roots, table_type="patrol_legs")


def test_read_names_a_bound_table_the_data_set_does_not_have(data_set_root):
    roots = json.dumps({"patrol_legs": {"url": data_set_root, "table_id": "legs_vehicles"}})
    with pytest.raises(DataSetSchemaError, match=r"no table 'legs_vehicles'.*available: patrol_events, patrol_legs"):
        read_data_set_table(data_set_roots=roots, table_type="patrol_legs")


def test_a_manifest_from_before_table_ids_still_reads(data_set_root):
    """Keyed by type name, entries without table_type: each id is its type."""
    p = Path(data_set_root) / "data_sets" / "manifest.json"
    m = json.loads(p.read_text())
    for entry in m["tables"].values():
        del entry["table_type"]
    p.write_text(json.dumps(m))
    df = read_data_set_table(data_set_roots=json.dumps({"patrol_legs": data_set_root}), table_type="patrol_legs")
    assert len(df) == 2


@pytest.mark.parametrize("bound", [None, 3, {"table_id": "patrol_legs"}, {"url": None}, {"url": "x", "table_id": 1}])
def test_read_refuses_a_binding_that_is_not_a_root_or_a_url_and_table(bound):
    with pytest.raises(DataSetSchemaError, match="patrol_legs: a binding is a results root or"):
        read_data_set_table(data_set_roots=json.dumps({"patrol_legs": bound}), table_type="patrol_legs")
