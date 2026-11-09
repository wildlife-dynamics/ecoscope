import pandas as pd
import pytest

from ecoscope.platform.data_sets import TABLE_TYPES, DataSetManifest, DataSetSchemaError, read_table, table_to_frame
from ecoscope.platform.tasks.data_sets import gather_data_sets


def test_gather_writes_every_table_and_manifest(tmp_path, patrol_events_gdf, patrol_legs_df):
    manifest = gather_data_sets(
        data_set="patrols",
        tables={"patrol_events": patrol_events_gdf, "patrol_legs": patrol_legs_df},
        root_path=str(tmp_path),
    )
    assert set(manifest.tables) == {"patrol_events", "patrol_legs"}
    assert manifest.tables["patrol_legs"].rows == 2 and manifest.tables["patrol_legs"].schema_version == "1"
    assert manifest.tables["patrol_legs"].path == "data_sets/patrol_legs.arrow"
    assert (tmp_path / "data_sets" / "patrol_events.arrow").exists()
    assert DataSetManifest.model_validate_json((tmp_path / "data_sets" / "manifest.json").read_text()) == manifest


def test_gather_writes_an_empty_table(tmp_path, patrol_events_gdf):
    manifest = gather_data_sets(
        data_set="patrols", tables={"patrol_events": patrol_events_gdf.iloc[0:0]}, root_path=str(tmp_path)
    )
    assert manifest.tables["patrol_events"].rows == 0


def test_gather_rejects_unknown_table_type(tmp_path, patrol_legs_df):
    with pytest.raises(DataSetSchemaError, match="unknown table type 'legs'"):
        gather_data_sets(data_set="p", tables={"legs": patrol_legs_df}, root_path=str(tmp_path))
    assert not (tmp_path / "data_sets").exists()


def test_gather_rejects_unsupported_scheme_before_writing(patrol_legs_df):
    with pytest.raises(ValueError):
        gather_data_sets(data_set="p", tables={"patrol_legs": patrol_legs_df}, root_path="s3://bucket/run")


def test_gather_is_discoverable_by_the_registry():
    from wt_registry import get_registry

    assert any(e.function_name == "gather_data_sets" for e in get_registry().values())


def test_gather_validates_every_table_before_writing_any(tmp_path, patrol_legs_df):
    with pytest.raises(DataSetSchemaError):
        gather_data_sets(
            data_set="p",
            tables={"patrol_legs": patrol_legs_df, "patrol_events": pd.DataFrame({"x": [1]})},
            root_path=str(tmp_path),
        )
    assert not list(tmp_path.glob("data_sets/*.arrow"))


def test_gather_writes_a_columnless_empty_spatial_frame_with_crs(tmp_path):
    gather_data_sets(data_set="p", tables={"patrol_events": pd.DataFrame()}, root_path=str(tmp_path))
    frame = table_to_frame(read_table(str(tmp_path), "patrol_events", "1"), "patrol_events")
    assert len(frame) == 0 and frame.crs == "EPSG:4326"


def test_gather_writes_a_columnless_empty_frame_from_the_registry_schema(tmp_path):
    manifest = gather_data_sets(data_set="p", tables={"patrol_legs": pd.DataFrame()}, root_path=str(tmp_path))
    assert manifest.tables["patrol_legs"].rows == 0
    assert (tmp_path / "data_sets" / "patrol_legs.arrow").exists()
    frame = table_to_frame(read_table(str(tmp_path), "patrol_legs", "1"), "patrol_legs")
    assert len(frame) == 0
    assert set(TABLE_TYPES["patrol_legs"].schema.names) <= set(frame.columns)


def test_gather_writes_a_skipped_step_as_an_empty_table(tmp_path, patrol_legs_df):
    from wt_task import SKIP_SENTINEL, task

    manifest = (
        task(gather_data_sets)
        .validate()
        .call(
            data_set="patrols",
            tables={"patrol_events": SKIP_SENTINEL, "patrol_legs": patrol_legs_df},
            root_path=str(tmp_path),
        )
    )
    assert manifest.tables["patrol_events"].rows == 0
    assert manifest.tables["patrol_legs"].rows == 2
    frame = table_to_frame(read_table(str(tmp_path), "patrol_events", "1"), "patrol_events")
    assert len(frame) == 0
    assert set(TABLE_TYPES["patrol_events"].schema.names) <= set(frame.columns)
    assert frame.crs == "EPSG:4326"
    legs = table_to_frame(read_table(str(tmp_path), "patrol_legs", "1"), "patrol_legs")
    assert legs["patrol_id"].tolist() == patrol_legs_df["patrol_id"].tolist()
