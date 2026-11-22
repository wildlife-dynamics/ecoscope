import json

import geopandas as gpd  # type: ignore[import-untyped]
import numpy as np
import pandas as pd
import pytest
from pydantic import TypeAdapter, ValidationError
from shapely.geometry import Point

from ecoscope.platform.tasks.data_sets import apply_processing_steps
from ecoscope.platform.tasks.data_sets._steps import AggregateStep, FilterStep, ProcessingStep


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


def steps_of(*items):
    return TypeAdapter(list[ProcessingStep]).validate_python(list(items))


def apply(df, tmp_path, *items, table_id="events"):
    return apply_processing_steps(df=df, table_id=table_id, root_path=str(tmp_path), steps=steps_of(*items))


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


def test_filter_is_any_of_keeps_matching_rows(tmp_path, events):
    result = apply(
        events, tmp_path, {"type": "filter", "column": "event_type", "op": "is_any_of", "values": ["carcass", "snare"]}
    )
    assert isinstance(result, gpd.GeoDataFrame) and result.crs == "EPSG:4326"
    assert result["event_type"].tolist() == ["carcass", "snare", "carcass"]
    written = preview(tmp_path)
    assert [s["rows"] for s in written["steps"]] == [5, 3]
    assert written["steps"][1]["error"] is None
    assert [c["name"] for c in written["steps"][1]["columns"]] == list(events.columns)


@pytest.mark.parametrize(
    "column, op, values, kept",
    [
        ("event_type", "is_any_of", ["snare", "sighting"], [1, 3]),
        ("event_type", "is_none_of", ["carcass"], [1, 3, 4]),
        ("event_type", "eq", ["snare"], [1]),
        ("event_type", "ne", ["carcass"], [1, 3, 4]),
        ("dist_meters", "gt", [30], [0, 1]),
        ("dist_meters", "ge", ["30"], [0, 1, 2]),
        ("dist_meters", "lt", [30], [4]),
        ("dist_meters", "le", [30.0], [2, 4]),
        ("dist_meters", "is_empty", [], [3]),
        ("dist_meters", "is_not_empty", [], [0, 1, 2, 4]),
        ("event_type", "is_empty", [], [4]),
        ("geometry", "is_not_empty", [], [0, 1, 2, 3]),
        ("time", "ge", ["2025-01-13"], [2, 3, 4]),
        ("time", "lt", ["2025-01-08T09:30:00+00:00"], [0]),
        ("time", "eq", ["2025-01-08T12:30:00+03:00"], [1]),
        ("is_collected", "eq", ["true"], [0, 2, 4]),
        ("is_collected", "is_any_of", [False], [1, 3]),
        ("event_type", "gt", ["sighting"], [1]),
    ],
)
def test_filter_ops(tmp_path, events, column, op, values, kept):
    result = apply(events, tmp_path, {"type": "filter", "column": column, "op": op, "values": values})
    assert result.index.tolist() == kept


def test_filter_value_not_coercible_fails_naming_column_and_value(tmp_path, events):
    with pytest.raises(ValueError, match=r"'abc'.*number column 'dist_meters'"):
        apply(events, tmp_path, {"type": "filter", "column": "dist_meters", "op": "gt", "values": ["abc"]})
    last = preview(tmp_path)["steps"][-1]
    assert last["rows"] is None and "'abc'" in last["error"] and "'dist_meters'" in last["error"]


@pytest.mark.parametrize(
    "column, value, kind",
    [("time", "yesterday-ish", "date"), ("time", 5, "date"), ("is_collected", "maybe", "boolean")],
)
def test_filter_value_not_coercible_names_the_column_type(tmp_path, events, column, value, kind):
    with pytest.raises(ValueError, match=rf"{value!r}.*{kind} column '{column}'"):
        apply(events, tmp_path, {"type": "filter", "column": column, "op": "eq", "values": [value]})


def test_filter_comparison_without_a_value_fails(tmp_path, events):
    with pytest.raises(ValueError, match=r"'gt'.*'dist_meters'.*needs a value"):
        apply(events, tmp_path, {"type": "filter", "column": "dist_meters", "op": "gt"})


def test_aggregate_by_week_and_group_counts_and_sums(tmp_path, events):
    result = apply(
        events,
        tmp_path,
        {"type": "filter", "column": "event_type", "op": "is_not_empty"},
        {
            "type": "aggregate",
            "group_by": ["event_type"],
            "time_bucket": {"column": "time", "interval": "week"},
            "measures": [{"op": "count"}, {"op": "sum", "column": "dist_meters"}],
        },
    )
    assert list(result.columns) == ["time", "event_type", "count", "sum_dist_meters"]
    assert str(result["time"].dtype) == "datetime64[ns, UTC]"
    assert result.to_dict("records") == [
        {"time": pd.Timestamp("2025-01-06", tz="UTC"), "event_type": "carcass", "count": 1, "sum_dist_meters": 100.0},
        {"time": pd.Timestamp("2025-01-06", tz="UTC"), "event_type": "snare", "count": 1, "sum_dist_meters": 250.5},
        {"time": pd.Timestamp("2025-01-13", tz="UTC"), "event_type": "carcass", "count": 1, "sum_dist_meters": 30.0},
        {"time": pd.Timestamp("2025-01-13", tz="UTC"), "event_type": "sighting", "count": 1, "sum_dist_meters": 0.0},
    ]
    written = preview(tmp_path)
    assert [s["rows"] for s in written["steps"]] == [5, 4, 4]
    assert [(c["name"], c["type"]) for c in written["steps"][2]["columns"]] == [
        ("time", "date"),
        ("event_type", "text"),
        ("count", "number"),
        ("sum_dist_meters", "number"),
    ]
    assert written["steps"][2]["sample"][0] == {
        "time": "2025-01-06T00:00:00+00:00",
        "event_type": "carcass",
        "count": 1,
        "sum_dist_meters": 100.0,
    }


@pytest.mark.parametrize(
    "interval, starts",
    [
        ("day", ["2025-01-06", "2025-01-08", "2025-01-13", "2025-01-14", "2025-01-15"]),
        ("week", ["2025-01-06", "2025-01-13"]),
        ("month", ["2025-01-01"]),
        ("year", ["2025-01-01"]),
    ],
)
def test_aggregate_time_bucket_intervals_start_in_utc(tmp_path, events, interval, starts):
    events["time"] = events["time"].dt.tz_convert("Africa/Nairobi")
    result = apply(events, tmp_path, {"type": "aggregate", "time_bucket": {"column": "time", "interval": interval}})
    assert list(result.columns) == ["time", "count"]
    assert result["time"].tolist() == [pd.Timestamp(s, tz="UTC") for s in starts]
    assert result["count"].sum() == 5


def test_aggregate_measures_without_groups_give_one_row(tmp_path, events):
    result = apply(
        events,
        tmp_path,
        {
            "type": "aggregate",
            "measures": [
                {"op": "count"},
                {"op": "mean", "column": "dist_meters"},
                {"op": "min", "column": "dist_meters"},
                {"op": "max", "column": "time"},
                {"op": "median", "column": "dist_meters"},
                {"op": "nunique", "column": "event_type"},
            ],
        },
    )
    assert list(result.columns) == [
        "count",
        "mean_dist_meters",
        "min_dist_meters",
        "max_time",
        "median_dist_meters",
        "nunique_event_type",
    ]
    assert result.iloc[0].tolist() == [5, 96.375, 5.0, pd.Timestamp("2025-01-15 12:00", tz="UTC"), 65.0, 3]


def test_aggregate_group_by_alone_counts_rows_and_keeps_null_groups(tmp_path, events):
    result = apply(events, tmp_path, {"type": "aggregate", "group_by": ["event_type"]})
    assert list(result.columns) == ["event_type", "count"]
    assert result["event_type"].tolist()[:3] == ["carcass", "sighting", "snare"] and pd.isna(result["event_type"][3])
    assert result["count"].tolist() == [2, 1, 1, 1]


def test_aggregate_drops_geometry_and_returns_plain_dataframe(tmp_path, events):
    result = apply(events, tmp_path, {"type": "aggregate", "group_by": ["event_type"]})
    assert type(result) is pd.DataFrame
    assert "geometry" not in result.columns
    assert all(c["type"] != "geometry" for c in preview(tmp_path)["steps"][1]["columns"])


def test_step_on_missing_column_writes_error_entry_then_raises(tmp_path, events):
    with pytest.raises(ValueError, match="column 'dist_meters' not found after step 1"):
        apply(
            events,
            tmp_path,
            {"type": "aggregate", "group_by": ["event_type"]},
            {"type": "filter", "column": "dist_meters", "op": "gt", "values": [1]},
            {"type": "filter", "column": "count", "op": "gt", "values": [1]},
        )
    written = preview(tmp_path)
    assert [s["rows"] for s in written["steps"]] == [5, 4, None]
    assert written["steps"][2] == {
        "rows": None,
        "columns": [],
        "sample": [],
        "error": "column 'dist_meters' not found after step 1",
    }


def test_first_step_on_missing_column_names_the_source(tmp_path, events):
    with pytest.raises(ValueError, match="column 'nope' not found in the source table"):
        apply(events, tmp_path, {"type": "filter", "column": "nope", "op": "is_empty"})


@pytest.mark.parametrize(
    "step, message",
    [
        ({"type": "aggregate", "measures": [{"op": "sum", "column": "event_type"}]}, "cannot sum text column"),
        ({"type": "aggregate", "time_bucket": {"column": "event_type", "interval": "day"}}, "not a date column"),
        ({"type": "aggregate", "group_by": ["geometry"]}, "cannot group by geometry column 'geometry'"),
        ({"type": "aggregate", "group_by": ["time"], "time_bucket": {"column": "time", "interval": "day"}}, "twice"),
        ({"type": "filter", "column": "geometry", "op": "eq", "values": ["x"]}, "geometry column 'geometry'"),
    ],
)
def test_step_that_cannot_run_records_its_error(tmp_path, events, step, message):
    with pytest.raises(ValueError, match=message):
        apply(events, tmp_path, step)
    assert message in preview(tmp_path)["steps"][-1]["error"]


def test_empty_frame_with_steps_writes_zero_rows(tmp_path, events):
    steps = [
        {"type": "filter", "column": "event_type", "op": "eq", "values": ["snare"]},
        {
            "type": "aggregate",
            "group_by": ["event_type"],
            "time_bucket": {"column": "time", "interval": "month"},
            "measures": [{"op": "count"}, {"op": "mean", "column": "dist_meters"}],
        },
        {"type": "filter", "column": "count", "op": "gt", "values": [0]},
    ]
    result = apply(events.iloc[0:0], tmp_path, *steps)
    assert len(result) == 0
    assert list(result.columns) == ["time", "event_type", "count", "mean_dist_meters"]
    written = preview(tmp_path)
    assert [s["rows"] for s in written["steps"]] == [0, 0, 0, 0]
    assert [c["name"] for c in written["steps"][0]["columns"]] == list(events.columns)
    assert all(s["sample"] == [] for s in written["steps"])


def test_columnless_empty_frame_with_steps_writes_zero_rows(tmp_path):
    result = apply(
        pd.DataFrame(),
        tmp_path,
        {"type": "filter", "column": "event_type", "op": "eq", "values": ["snare"]},
        {"type": "aggregate", "time_bucket": {"column": "time", "interval": "day"}, "group_by": ["event_type"]},
        {"type": "filter", "column": "count", "op": "gt", "values": [0]},
    )
    assert len(result) == 0 and list(result.columns) == ["time", "event_type", "count"]
    written = preview(tmp_path)
    assert [s["rows"] for s in written["steps"]] == [0, 0, 0, 0]
    assert written["steps"][0]["columns"] == []
    assert [c["type"] for c in written["steps"][2]["columns"]] == ["date", "text", "number"]


def test_aggregate_without_groups_on_an_empty_frame_gives_no_rows(tmp_path, events):
    result = apply(events.iloc[0:0], tmp_path, {"type": "aggregate", "measures": [{"op": "count"}]})
    assert len(result) == 0 and list(result.columns) == ["count"]


def test_skipped_upstream_step_is_an_empty_frame(tmp_path):
    from wt_task import SKIP_SENTINEL, task

    result = (
        task(apply_processing_steps)
        .validate()
        .call(
            df=SKIP_SENTINEL,
            table_id="legs",
            root_path=str(tmp_path),
            steps=[{"type": "filter", "column": "patrol_type", "op": "eq", "values": ["foot"]}],
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
            steps=[{"type": "filter", "column": "event_type", "op": "eq", "values": ["snare"]}],
        )
    )
    assert isinstance(result, gpd.GeoDataFrame) and len(result) == 1


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


def test_step_models_validate_the_contract_shapes():
    steps = steps_of(
        {"type": "filter", "column": "event_type", "op": "is_empty"},
        {"type": "aggregate", "group_by": ["event_type"]},
    )
    assert isinstance(steps[0], FilterStep) and steps[0].values == []
    assert isinstance(steps[1], AggregateStep) and steps[1].time_bucket is None
    assert [m.model_dump() for m in steps[1].measures] == [{"op": "count", "column": None}]
    with pytest.raises(ValidationError, match="at least one"):
        steps_of({"type": "aggregate"})
    with pytest.raises(ValidationError, match="needs a column"):
        steps_of({"type": "aggregate", "measures": [{"op": "sum"}]})
    with pytest.raises(ValidationError):
        steps_of({"type": "sort", "column": "x"})
    with pytest.raises(ValidationError):
        steps_of({"type": "filter", "column": "x", "op": "contains"})
    with pytest.raises(ValidationError):
        steps_of({"type": "aggregate", "time_bucket": {"column": "time", "interval": "hour"}})


def test_steps_form_schema_is_a_plain_array_of_objects():
    from wt_registry.jsonschema import jsonschema_from_task_func

    schema = jsonschema_from_task_func(apply_processing_steps)
    assert schema["properties"]["steps"] == {
        "type": "array",
        "items": {"type": "object"},
        "default": [],
        "title": "Processing Steps",
        "description": "Filter and aggregate steps, applied in order.",
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
    assert not {"FilterStep", "AggregateStep", "TimeBucket", "Measure"} & set(schema.get("$defs", {}))
    assert not {"anyOf", "oneOf", "discriminator"} & set(keys({k: v for k, v in schema.items() if k != "$defs"}))


def test_steps_are_taken_as_dicts_or_models(tmp_path, events):
    as_dict = {"type": "filter", "column": "event_type", "op": "eq", "values": ["snare"]}
    aggregate = {"type": "aggregate", "group_by": ["event_type"]}
    result = apply_processing_steps(
        df=events, table_id="events", root_path=str(tmp_path), steps=[as_dict, AggregateStep(**aggregate)]
    )
    assert result.to_dict("records") == [{"event_type": "snare", "count": 1}]
    assert [s["rows"] for s in preview(tmp_path)["steps"]] == [5, 1, 1]


@pytest.mark.parametrize(
    "bad, message",
    [
        ({"type": "sort", "column": "x"}, "step 2: unknown step type 'sort'; a step is a 'filter' or an 'aggregate'"),
        ({"column": "x"}, "step 2: a step needs a type, 'filter' or 'aggregate'"),
        ({"type": "aggregate", "measures": [{"op": "sum"}]}, "step 2: measure 'sum' needs a column"),
        (
            {"type": "aggregate"},
            "step 2: an aggregate step needs at least one of group_by, time_bucket and measures",
        ),
        ({"type": "filter", "op": "eq"}, "step 2: column is required"),
        (
            {"type": "filter", "column": "x", "op": "contains"},
            "step 2: op: Input should be 'is_any_of', 'is_none_of', 'eq', 'ne', 'gt', 'ge', 'lt', 'le', "
            "'is_empty' or 'is_not_empty'",
        ),
        (
            {"type": "aggregate", "time_bucket": {"column": "time", "interval": "hour"}},
            "step 2: time_bucket.interval: Input should be 'day', 'week', 'month' or 'year'",
        ),
        ("filter", "step 2: a step is an object, not 'filter'"),
    ],
)
def test_invalid_step_is_recorded_at_its_position_then_raises(tmp_path, events, bad, message):
    from wt_task import task

    good = {"type": "filter", "column": "event_type", "op": "eq", "values": ["snare"]}
    with pytest.raises(ValueError) as raised:
        task(apply_processing_steps).validate().call(
            df=events, table_id="events", root_path=str(tmp_path), steps=[good, bad, good]
        )
    assert str(raised.value) == message
    written = preview(tmp_path)
    assert [s["rows"] for s in written["steps"]] == [5, 1, None]
    assert written["steps"][2] == {"rows": None, "columns": [], "sample": [], "error": message}


def test_apply_processing_steps_is_registered_no_cache():
    from wt_registry import get_registry

    entry = next(e for e in get_registry().values() if e.function_name == "apply_processing_steps")
    assert "no-cache" in entry.metadata.tags
