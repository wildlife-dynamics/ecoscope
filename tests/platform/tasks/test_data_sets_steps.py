import json

import geopandas as gpd  # type: ignore[import-untyped]
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Point

from ecoscope.platform.tasks.data_sets import apply_processing_steps
from ecoscope.platform.tasks.data_sets._steps import STEP_TASKS, ProcessingStep


@pytest.fixture
def events():
    return gpd.GeoDataFrame(
        {
            "event_type": ["carcass", "snare", "carcass", "sighting", None],
            "time": pd.to_datetime(
                ["2025-01-06 08:00", "2025-01-08 09:30", "2025-01-13 10:00", "2025-01-14 11:00", "2025-01-15 12:00"]
            ).tz_localize("UTC"),
            "dist_meters": [100.0, 250.5, 30.0, np.nan, 5.0],
            "is_collected": [True, False, True, False, True],
        },
        geometry=[Point(36.8, -1.3), Point(36.9, -1.4), Point(37.0, -1.5), Point(37.1, -1.6), None],
        crs="EPSG:4326",
    )


def step(task, **params):
    return {"task": task, "params": params}


def apply(df, tmp_path, *items, table_id="events"):
    return apply_processing_steps(df=df, table_id=table_id, root_path=str(tmp_path), steps=list(items))


def preview(tmp_path, table_id="events"):
    return json.loads((tmp_path / "data_sets" / "preview" / f"{table_id}.json").read_text())


def test_no_steps_returns_frame_and_writes_source_preview(tmp_path, events):
    result = apply_processing_steps(df=events, table_id="events", root_path=str(tmp_path))
    assert isinstance(result, gpd.GeoDataFrame) and result.equals(events)
    written = preview(tmp_path)
    assert written["table_id"] == "events"
    assert len(written["steps"]) == 1
    source = written["steps"][0]
    assert source["rows"] == 5
    assert source["columns"] == [
        {"name": "event_type", "type": "text", "values": ["carcass", "sighting", "snare"]},
        {"name": "time", "type": "date"},
        {"name": "dist_meters", "type": "number"},
        {"name": "is_collected", "type": "boolean", "values": ["false", "true"]},
        {"name": "geometry", "type": "geometry"},
    ]
    assert source["sample"][0] == {
        "event_type": "carcass",
        "time": "2025-01-06T08:00:00+00:00",
        "dist_meters": 100.0,
        "is_collected": True,
        "geometry": None,
    }
    assert source["sample"][3]["dist_meters"] is None and source["sample"][4]["event_type"] is None


def test_filter_step_keeps_matching_rows_and_the_geodataframe(tmp_path, events):
    result = apply(
        events, tmp_path, step("filter_df", column_name="event_type", op="is_any_of", values=["carcass", "snare"])
    )
    assert isinstance(result, gpd.GeoDataFrame) and result.crs == "EPSG:4326"
    assert result["event_type"].tolist() == ["carcass", "snare", "carcass"]
    written = preview(tmp_path)
    assert [s["rows"] for s in written["steps"]] == [5, 3]
    assert written["steps"][1]["error"] is None
    assert [c["name"] for c in written["steps"][1]["columns"]] == list(events.columns)


def test_summary_by_month_and_group(tmp_path, events):
    result = apply(
        events,
        tmp_path,
        step("filter_df", column_name="event_type", op="is_not_empty"),
        step("add_temporal_index", time_col="time", groupers=[{"temporal_index": "%Y-%m"}]),
        step(
            "summarize_df",
            groupby_cols=["TemporalGrouper_%Y-%m", "event_type"],
            summary_params=[
                {"display_name": "events", "aggregator": "count", "column": "time"},
                {"display_name": "distance", "aggregator": "sum", "column": "dist_meters"},
            ],
        ),
    )
    assert type(result) is pd.DataFrame
    assert result.to_dict("records") == [
        {"TemporalGrouper_%Y-%m": "2025-01", "event_type": "carcass", "events": 2.0, "distance": 130.0},
        {"TemporalGrouper_%Y-%m": "2025-01", "event_type": "sighting", "events": 1.0, "distance": 0.0},
        {"TemporalGrouper_%Y-%m": "2025-01", "event_type": "snare", "events": 1.0, "distance": 250.5},
    ]
    written = preview(tmp_path)
    assert [s["rows"] for s in written["steps"]] == [5, 4, 4, 3]
    assert [(c["name"], c["type"]) for c in written["steps"][3]["columns"]] == [
        ("TemporalGrouper_%Y-%m", "text"),
        ("event_type", "text"),
        ("events", "number"),
        ("distance", "number"),
    ]


def test_index_levels_a_step_adds_are_previewed_and_returned_as_columns(tmp_path, events):
    result = apply(events, tmp_path, step("add_temporal_index", time_col="time", groupers=[{"temporal_index": "%Y"}]))
    assert isinstance(result, gpd.GeoDataFrame)
    assert list(result.columns) == ["TemporalGrouper_%Y", *events.columns]
    assert result.index.names == [None]
    assert preview(tmp_path)["steps"][1]["columns"][0] == {
        "name": "TemporalGrouper_%Y",
        "type": "text",
        "values": ["2025"],
    }


def test_summary_without_groups_gives_one_row_and_drops_geometry(tmp_path, events):
    result = apply(
        events,
        tmp_path,
        step(
            "summarize_df",
            summary_params=[
                {"display_name": "types", "aggregator": "nunique", "column": "event_type"},
                {"display_name": "furthest", "aggregator": "max", "column": "dist_meters"},
            ],
        ),
    )
    assert type(result) is pd.DataFrame
    assert result.to_dict("records") == [{"types": 3.0, "furthest": 250.5}]
    assert all(c["type"] != "geometry" for c in preview(tmp_path)["steps"][1]["columns"])


def test_steps_chain_through_other_tasks(tmp_path, events):
    result = apply(
        events,
        tmp_path,
        step("fill_na", value=0, columns=["dist_meters"]),
        step("map_values", column_name="event_type", value_map={"snare": "Snare"}),
        step("convert_values_to_timezone", timezone="Africa/Nairobi", columns=["time"]),
        step("sort_values", column_name="dist_meters", ascending=False),
        step(
            "map_columns",
            drop_columns=["is_collected"],
            rename_columns=[{"original_name": "dist_meters", "new_name": "meters"}],
        ),
    )
    assert isinstance(result, gpd.GeoDataFrame)
    assert list(result.columns) == ["event_type", "time", "meters", "geometry"]
    assert result["meters"].tolist() == [250.5, 100.0, 30.0, 5.0, 0.0]
    assert result["event_type"].iloc[0] == "Snare"
    assert str(result["time"].dt.tz) == "Africa/Nairobi"
    assert [s["rows"] for s in preview(tmp_path)["steps"]] == [5] * 6


def test_a_step_does_not_change_the_frame_it_was_given(tmp_path, events):
    before = events.copy()
    apply(events, tmp_path, step("add_temporal_index", time_col="time", groupers=[{"temporal_index": "%Y"}]))
    assert events.equals(before) and events.index.names == [None]


def test_every_step_task_takes_the_table_as_df():
    import inspect

    assert "apply_sql_query" not in STEP_TASKS
    for name, task in STEP_TASKS.items():
        assert next(iter(inspect.signature(task).parameters)) == "df", name


def test_step_that_cannot_run_writes_error_entry_then_raises(tmp_path, events):
    message = "step 2: filter_df: column 'dist_meters' not found"
    with pytest.raises(ValueError) as raised:
        apply(
            events,
            tmp_path,
            step("map_columns", retain_columns=["event_type"]),
            step("filter_df", column_name="dist_meters", op="gt", value="1"),
            step("sort_values", column_name="event_type"),
        )
    assert str(raised.value) == message
    written = preview(tmp_path)
    assert [s["rows"] for s in written["steps"]] == [5, 5, None]
    assert written["steps"][2] == {"rows": None, "columns": [], "sample": [], "error": message}


def test_task_error_is_recorded_with_its_step_and_task(tmp_path, events):
    message = "step 1: filter_df: cannot compare value 'abc' with number column 'dist_meters'"
    with pytest.raises(ValueError) as raised:
        apply(events, tmp_path, step("filter_df", column_name="dist_meters", op="gt", value="abc"))
    assert str(raised.value) == message
    assert preview(tmp_path)["steps"][-1]["error"] == message


def test_table_without_rows_goes_through_unchanged(tmp_path, events):
    steps = [
        step("filter_df", column_name="event_type", op="equal", value="snare"),
        step(
            "summarize_df",
            groupby_cols=["event_type"],
            summary_params=[{"display_name": "events", "aggregator": "count", "column": "time"}],
        ),
    ]
    result = apply(events.iloc[0:0], tmp_path, *steps)
    assert len(result) == 0 and list(result.columns) == list(events.columns)
    written = preview(tmp_path)
    assert [s["rows"] for s in written["steps"]] == [0, 0, 0]
    assert all(s["sample"] == [] for s in written["steps"])

    result = apply(events, tmp_path, step("filter_df", column_name="event_type", op="equal", value="none"), steps[1])
    assert len(result) == 0 and list(result.columns) == list(events.columns)


def test_table_without_rows_still_has_its_steps_validated(tmp_path, events):
    with pytest.raises(ValueError, match="step 1: unknown task 'explode'"):
        apply(events.iloc[0:0], tmp_path, step("explode", column_name="event_type"))


def test_skipped_upstream_step_is_an_empty_frame(tmp_path):
    from wt_task import SKIP_SENTINEL, task

    result = (
        task(apply_processing_steps)
        .validate()
        .call(
            df=SKIP_SENTINEL,
            table_id="legs",
            root_path=str(tmp_path),
            steps=[step("filter_df", column_name="patrol_type", op="equal", value="foot")],
        )
    )
    assert isinstance(result, pd.DataFrame) and result.empty and len(result.columns) == 0
    assert [s["rows"] for s in preview(tmp_path, "legs")["steps"]] == [0, 0]


def test_validated_call_takes_step_dicts_and_keeps_the_geodataframe(tmp_path, events):
    from wt_task import task

    result = (
        task(apply_processing_steps)
        .validate()
        .call(
            df=events,
            table_id="events",
            root_path=str(tmp_path),
            steps=[step("filter_df", column_name="event_type", op="equal", value="snare")],
        )
    )
    assert isinstance(result, gpd.GeoDataFrame) and len(result) == 1


def test_steps_are_taken_as_dicts_or_models(tmp_path, events):
    as_dict = step("filter_df", column_name="event_type", op="equal", value="snare")
    as_model = ProcessingStep(task="sort_values", params={"column_name": "time"})
    result = apply_processing_steps(df=events, table_id="events", root_path=str(tmp_path), steps=[as_dict, as_model])
    assert result["event_type"].tolist() == ["snare"]
    assert [s["rows"] for s in preview(tmp_path)["steps"]] == [5, 1, 1]


def test_no_params_is_the_same_as_empty_params(tmp_path, events):
    result = apply(events, tmp_path, {"task": "map_columns"})
    assert list(result.columns) == list(events.columns)


TASKS = ", ".join(STEP_TASKS)


@pytest.mark.parametrize(
    "bad, message",
    [
        (
            {"task": "apply_sql_query", "params": {"query": "SELECT 1"}},
            f"unknown task 'apply_sql_query'; a step runs one of: {TASKS}",
        ),
        ({"params": {"column_name": "x"}}, f"a step needs a task, one of: {TASKS}"),
        ({"task": "filter_df", "params": {"op": "equal"}}, "filter_df: column_name is required"),
        ({"task": "filter_df", "params": {"column_name": "x", "op": "equal", "df": 1}}, "filter_df: no parameter 'df'"),
        (
            {"task": "filter_df", "params": {"column_name": "x", "op": "contains"}},
            "filter_df: op: Input should be 'equal', 'ge', 'gt', 'le', 'lt', 'ne', 'is_any_of', 'is_none_of', "
            "'is_empty' or 'is_not_empty'",
        ),
        (
            {"task": "add_temporal_index", "params": {"time_col": "time", "groupers": [{"temporal_index": "%W"}]}},
            "add_temporal_index: groupers",
        ),
        ({"task": "sort_values", "params": ["time"]}, "sort_values: params is an object, not ['time']"),
        (
            {"task": "sort_values", "column_name": "time"},
            "sort_values: a step has a task and its params, not 'column_name'",
        ),
        ("filter_df", "a step is an object, not 'filter_df'"),
    ],
)
def test_invalid_step_is_recorded_at_its_position_then_raises(tmp_path, events, bad, message):
    from wt_task import task

    good = step("filter_df", column_name="event_type", op="equal", value="snare")
    with pytest.raises(ValueError) as raised:
        task(apply_processing_steps).validate().call(
            df=events, table_id="events", root_path=str(tmp_path), steps=[good, bad, good]
        )
    assert str(raised.value).startswith(f"step 2: {message}")
    written = preview(tmp_path)
    assert [s["rows"] for s in written["steps"]] == [5, 1, None]
    assert written["steps"][2] == {"rows": None, "columns": [], "sample": [], "error": str(raised.value)}


def test_steps_form_schema_is_a_plain_array_of_objects():
    from wt_registry.jsonschema import jsonschema_from_task_func

    schema = jsonschema_from_task_func(apply_processing_steps)
    assert schema["properties"]["steps"] == {
        "type": "array",
        "items": {"type": "object"},
        "default": [],
        "title": "Processing Steps",
        "description": "Task calls applied to the table in order, each `{task, params}`.",
    }

    def keys(node):
        if isinstance(node, dict):
            yield from node
            for value in node.values():
                yield from keys(value)
        elif isinstance(node, list):
            for value in node:
                yield from keys(value)

    assert not {"anyOf", "oneOf", "allOf", "discriminator", "$ref"} & set(keys(schema["properties"]["steps"]))
    assert "ProcessingStep" not in schema.get("$defs", {})
    assert not {"anyOf", "oneOf", "discriminator"} & set(keys({k: v for k, v in schema.items() if k != "$defs"}))


def test_preview_values_only_for_low_cardinality_text(tmp_path):
    df = pd.DataFrame(
        {
            "fifty": [f"v{i:02d}" for i in range(50)] + ["v00"],
            "fifty_one": [f"v{i:02d}" for i in range(51)],
            "number": range(51),
        }
    )
    apply(df, tmp_path)
    columns = {c["name"]: c for c in preview(tmp_path)["steps"][0]["columns"]}
    assert columns["fifty"]["values"] == [f"v{i:02d}" for i in range(50)]
    assert "values" not in columns["fifty_one"]
    assert "values" not in columns["number"]


def test_preview_sample_capped_at_100_and_json_safe(tmp_path):
    n = 150
    df = gpd.GeoDataFrame(
        {
            "name": [f"n{i}" for i in range(n)],
            "time": pd.date_range("2025-01-01", periods=n, freq="h", tz="Africa/Nairobi"),
            "naive": pd.date_range("2025-01-01", periods=n, freq="h"),
            "value": [np.nan, np.inf] + [1.5] * (n - 2),
            "whole": np.arange(n, dtype="int64"),
            "details": [{"a": [1, 2]}, [1, 2], np.array([1, 2])] + [None] * (n - 3),
            "mixed": [1, "one", 2.5] + [None] * (n - 3),
            "delta": pd.to_timedelta(range(n), unit="s"),
            "category": pd.Categorical(["a", "b"] * (n // 2)),
        },
        geometry=[Point(i, 0) for i in range(n)],
        crs="EPSG:4326",
    )
    apply(df, tmp_path)
    text = (tmp_path / "data_sets" / "preview" / "events.json").read_text()
    assert "NaN" not in text and "Infinity" not in text
    source = json.loads(text)["steps"][0]
    assert source["rows"] == n and len(source["sample"]) == 100
    assert {c["name"]: c["type"] for c in source["columns"]} == {
        "name": "text",
        "time": "date",
        "naive": "date",
        "value": "number",
        "whole": "number",
        "details": "other",
        "mixed": "other",
        "delta": "other",
        "category": "text",
        "geometry": "geometry",
    }
    first, second, third = source["sample"][:3]
    assert first == {
        "name": "n0",
        "time": "2024-12-31T21:00:00+00:00",
        "naive": "2025-01-01T00:00:00",
        "value": None,
        "whole": 0,
        "details": None,
        "mixed": 1,
        "delta": "0 days 00:00:00",
        "category": "a",
        "geometry": None,
    }
    assert second["value"] is None and second["details"] is None and second["mixed"] == "one"
    assert third["value"] == 1.5 and third["details"] is None and third["mixed"] == 2.5


def test_unusable_table_id_is_refused_before_writing(tmp_path, events):
    with pytest.raises(ValueError, match="table id"):
        apply(events, tmp_path, table_id="../events")
    assert not list(tmp_path.rglob("*.json"))


def test_apply_processing_steps_is_registered_no_cache():
    from wt_registry import get_registry

    entry = next(e for e in get_registry().values() if e.function_name == "apply_processing_steps")
    assert "no-cache" in entry.metadata.tags
