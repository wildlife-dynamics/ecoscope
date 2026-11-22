import geopandas as gpd  # type: ignore[import-untyped]
import pandas as pd
import pytest

from ecoscope.platform.data_sets import (
    TABLE_TYPES,
    DataSetManifest,
    DataSetSchemaError,
    read_table,
    table_filename,
    table_to_frame,
)
from ecoscope.platform.tasks.data_sets import gather_data_sets


def typed(tables):
    """Each frame as a table named after its type."""
    return {k: {"table_type": k, "frame": v} for k, v in tables.items()}


def test_gather_writes_every_table_and_manifest(tmp_path, patrol_events_gdf, patrol_legs_df):
    manifest = gather_data_sets(
        data_set="patrols",
        tables=typed({"patrol_events": patrol_events_gdf, "patrol_legs": patrol_legs_df}),
        root_path=str(tmp_path),
    )
    assert set(manifest.tables) == {"patrol_events", "patrol_legs"}
    assert manifest.tables["patrol_legs"].rows == 2 and manifest.tables["patrol_legs"].schema_version == "1"
    assert manifest.tables["patrol_legs"].path == "data_sets/patrol_legs.arrow"
    assert (tmp_path / "data_sets" / "patrol_events.arrow").exists()
    assert DataSetManifest.model_validate_json((tmp_path / "data_sets" / "manifest.json").read_text()) == manifest


def test_gather_writes_an_empty_table(tmp_path, patrol_events_gdf):
    manifest = gather_data_sets(
        data_set="patrols", tables=typed({"patrol_events": patrol_events_gdf.iloc[0:0]}), root_path=str(tmp_path)
    )
    assert manifest.tables["patrol_events"].rows == 0


def test_gather_rejects_unknown_table_type(tmp_path, patrol_legs_df):
    with pytest.raises(DataSetSchemaError, match="unknown table type 'legs'"):
        gather_data_sets(data_set="p", tables=typed({"legs": patrol_legs_df}), root_path=str(tmp_path))
    assert not (tmp_path / "data_sets").exists()


def test_gather_rejects_unsupported_scheme_before_writing(patrol_legs_df):
    with pytest.raises(ValueError):
        gather_data_sets(data_set="p", tables=typed({"patrol_legs": patrol_legs_df}), root_path="s3://bucket/run")


def test_gather_is_discoverable_by_the_registry():
    from wt_registry import get_registry

    assert any(e.function_name == "gather_data_sets" for e in get_registry().values())


def test_gather_validates_every_table_before_writing_any(tmp_path, patrol_legs_df):
    with pytest.raises(DataSetSchemaError):
        gather_data_sets(
            data_set="p",
            tables=typed({"patrol_legs": patrol_legs_df, "patrol_events": pd.DataFrame({"x": [1]})}),
            root_path=str(tmp_path),
        )
    assert not list(tmp_path.glob("data_sets/*.arrow"))


def test_gather_writes_a_columnless_empty_spatial_frame_with_crs(tmp_path):
    gather_data_sets(data_set="p", tables=typed({"patrol_events": pd.DataFrame()}), root_path=str(tmp_path))
    frame = table_to_frame(
        read_table(str(tmp_path), "patrol_events", "1", table_filename("patrol_events")), "patrol_events"
    )
    assert len(frame) == 0 and frame.crs == "EPSG:4326"


def test_gather_writes_a_columnless_empty_frame_from_the_registry_schema(tmp_path):
    manifest = gather_data_sets(data_set="p", tables=typed({"patrol_legs": pd.DataFrame()}), root_path=str(tmp_path))
    assert manifest.tables["patrol_legs"].rows == 0
    assert (tmp_path / "data_sets" / "patrol_legs.arrow").exists()
    frame = table_to_frame(read_table(str(tmp_path), "patrol_legs", "1", table_filename("patrol_legs")), "patrol_legs")
    assert len(frame) == 0
    assert set(TABLE_TYPES["patrol_legs"].schema.names) <= set(frame.columns)


def test_gather_writes_a_skipped_step_as_an_empty_table(tmp_path, patrol_legs_df):
    from wt_task import SKIP_SENTINEL, task

    manifest = (
        task(gather_data_sets)
        .validate()
        .call(
            data_set="patrols",
            tables=typed({"patrol_events": SKIP_SENTINEL, "patrol_legs": patrol_legs_df}),
            root_path=str(tmp_path),
        )
    )
    assert manifest.tables["patrol_events"].rows == 0
    assert manifest.tables["patrol_legs"].rows == 2
    frame = table_to_frame(
        read_table(str(tmp_path), "patrol_events", "1", table_filename("patrol_events")), "patrol_events"
    )
    assert len(frame) == 0
    assert set(TABLE_TYPES["patrol_events"].schema.names) <= set(frame.columns)
    assert frame.crs == "EPSG:4326"
    legs = table_to_frame(read_table(str(tmp_path), "patrol_legs", "1", table_filename("patrol_legs")), "patrol_legs")
    assert legs["patrol_id"].tolist() == patrol_legs_df["patrol_id"].tolist()


def test_two_tables_of_one_type_are_kept_apart_by_id(tmp_path, patrol_legs_df):
    manifest = gather_data_sets(
        data_set="patrols",
        tables={
            "legs_rangers": {"table_type": "patrol_legs", "frame": patrol_legs_df},
            "legs_vehicles": {"table_type": "patrol_legs", "frame": patrol_legs_df.iloc[:1]},
        },
        root_path=str(tmp_path),
    )
    assert {i: (t.table_type, t.path, t.rows) for i, t in manifest.tables.items()} == {
        "legs_rangers": ("patrol_legs", "data_sets/legs_rangers.arrow", len(patrol_legs_df)),
        "legs_vehicles": ("patrol_legs", "data_sets/legs_vehicles.arrow", 1),
    }
    assert sorted(f.name for f in tmp_path.glob("data_sets/*.arrow")) == [
        "legs_rangers.arrow",
        "legs_vehicles.arrow",
    ]


@pytest.mark.parametrize("table_id", ["Legs", "legs/other", "../legs", ""])
def test_gather_refuses_a_table_id_that_is_not_a_plain_name(tmp_path, patrol_legs_df, table_id):
    with pytest.raises(DataSetSchemaError, match="table id"):
        gather_data_sets(
            data_set="p",
            tables={table_id: {"table_type": "patrol_legs", "frame": patrol_legs_df}},
            root_path=str(tmp_path),
        )
    assert not list(tmp_path.glob("data_sets/*"))


def test_gather_writes_generic_when_allowed_and_schema_mismatches(tmp_path, patrol_events_gdf, patrol_legs_df):
    counts = patrol_legs_df.groupby("patrol_type", as_index=False).size()
    manifest = gather_data_sets(
        data_set="patrols",
        tables={
            "events": {
                "table_type": "patrol_events",
                "frame": patrol_events_gdf.drop(columns="patrol_id").assign(details=[{"a": 1}, [1, 2], None]),
                "allow_generic": True,
            },
            "legs": {"table_type": "patrol_legs", "frame": counts, "allow_generic": True},
        },
        root_path=str(tmp_path),
    )
    assert {i: (t.table_type, t.schema_version, t.path, t.rows) for i, t in manifest.tables.items()} == {
        "events": ("table", TABLE_TYPES["table"].schema_version, "data_sets/events.arrow", 3),
        "legs": ("table", TABLE_TYPES["table"].schema_version, "data_sets/legs.arrow", 2),
    }
    assert DataSetManifest.model_validate_json((tmp_path / "data_sets" / "manifest.json").read_text()) == manifest

    legs = table_to_frame(read_table(str(tmp_path), "table", "1", "data_sets/legs.arrow"), "table")
    assert type(legs) is pd.DataFrame
    pd.testing.assert_frame_equal(legs, counts)

    events = table_to_frame(read_table(str(tmp_path), "table", "1", "data_sets/events.arrow"), "table")
    assert isinstance(events, gpd.GeoDataFrame) and events.crs == "EPSG:4326"
    assert events.geometry.equals(patrol_events_gdf.geometry.reset_index(drop=True))
    assert events["details"].tolist() == [{"a": 1}, [1, 2], None]


def test_gather_still_fails_on_mismatch_without_allow_generic(tmp_path, patrol_legs_df):
    counts = patrol_legs_df.groupby("patrol_type", as_index=False).size()
    for entry in (
        {"table_type": "patrol_legs", "frame": counts},
        {"table_type": "patrol_legs", "frame": counts, "allow_generic": False},
    ):
        with pytest.raises(DataSetSchemaError, match="patrol_legs.*failed validation; missing fields"):
            gather_data_sets(data_set="p", tables={"legs": entry}, root_path=str(tmp_path))
    assert not list(tmp_path.glob("data_sets/*"))


def test_gather_keeps_declared_type_when_schema_matches_with_allow_generic(tmp_path, patrol_events_gdf, patrol_legs_df):
    manifest = gather_data_sets(
        data_set="patrols",
        tables={
            "events": {"table_type": "patrol_events", "frame": patrol_events_gdf, "allow_generic": True},
            "legs": {"table_type": "patrol_legs", "frame": patrol_legs_df.iloc[0:0], "allow_generic": True},
            "tracks": {"table_type": "patrol_tracks", "frame": pd.DataFrame(), "allow_generic": True},
        },
        root_path=str(tmp_path),
    )
    assert {i: t.table_type for i, t in manifest.tables.items()} == {
        "events": "patrol_events",
        "legs": "patrol_legs",
        "tracks": "patrol_tracks",
    }


def test_gather_reports_the_declared_type_when_a_frame_cannot_be_stored_at_all(tmp_path, patrol_legs_df):
    frame = patrol_legs_df.assign(odd=[object(), object()])
    with pytest.raises(DataSetSchemaError, match="patrol_legs: cannot"):
        gather_data_sets(
            data_set="p",
            tables={"legs": {"table_type": "patrol_legs", "frame": frame, "allow_generic": True}},
            root_path=str(tmp_path),
        )


def test_gather_accepts_a_table_declared_generic(tmp_path, patrol_legs_df):
    manifest = gather_data_sets(
        data_set="p",
        tables={
            "anything": {"table_type": "table", "frame": patrol_legs_df},
            "nothing": typed({"table": pd.DataFrame()})["table"],
        },
        root_path=str(tmp_path),
    )
    assert {i: (t.table_type, t.rows) for i, t in manifest.tables.items()} == {
        "anything": ("table", 2),
        "nothing": ("table", 0),
    }
    nothing = table_to_frame(read_table(str(tmp_path), "table", "1", "data_sets/nothing.arrow"), "table")
    assert nothing.empty and len(nothing.columns) == 0


def test_gather_takes_processed_frames_through_task_validation(tmp_path, patrol_events_gdf, patrol_legs_df):
    from wt_task import SKIP_SENTINEL, task

    from ecoscope.platform.tasks.data_sets import apply_processing_steps

    def processed(table_id, df, *steps):
        return (
            task(apply_processing_steps)
            .validate()
            .call(df=df, table_id=table_id, root_path=str(tmp_path), steps=list(steps))
        )

    aggregate = {"type": "aggregate", "group_by": ["patrol_type"]}
    frames = {
        "events": processed(
            "events", patrol_events_gdf, {"type": "filter", "column": "event_type", "op": "ne", "values": ["snare"]}
        ),
        "legs": processed("legs", patrol_legs_df, aggregate),
        "tracks": processed("tracks", SKIP_SENTINEL, aggregate),
    }
    types = {"events": "patrol_events", "legs": "patrol_legs", "tracks": "patrol_tracks"}
    manifest = (
        task(gather_data_sets)
        .validate()
        .call(
            data_set="patrols",
            tables={i: {"table_type": types[i], "frame": frames[i], "allow_generic": True} for i in types},
            root_path=str(tmp_path),
        )
    )
    assert {i: (t.table_type, t.rows) for i, t in manifest.tables.items()} == {
        "events": ("patrol_events", 2),
        "legs": ("table", 2),
        "tracks": ("table", 0),
    }
    assert sorted(p.name for p in tmp_path.glob("data_sets/preview/*.json")) == [
        "events.json",
        "legs.json",
        "tracks.json",
    ]
    tracks = table_to_frame(read_table(str(tmp_path), "table", "1", "data_sets/tracks.arrow"), "table")
    assert list(tracks.columns) == ["patrol_type", "count"] and len(tracks) == 0
