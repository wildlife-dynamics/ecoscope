import inspect
import json
import math
from functools import cache
from pathlib import Path
from typing import Annotated, Any, Callable, get_type_hints

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, ValidationError, WithJsonSchema
from wt_registry import register
from wt_task.skip import SkippedDependencyFallback, SkipSentinel

from ecoscope.platform.annotations import AnyDataFrameOrEmpty, AnyGeoDataFrameOrEmpty
from ecoscope.platform.data_sets import table_filename
from ecoscope.platform.serde import _get_path, _persist_text
from ecoscope.platform.tasks.analysis._summary import summarize_df
from ecoscope.platform.tasks.transformation._conversion import convert_values_to_timezone
from ecoscope.platform.tasks.transformation._filter import column_kind, filter_df
from ecoscope.platform.tasks.transformation._indexing import add_temporal_index
from ecoscope.platform.tasks.transformation._mapping import fill_na, map_columns, map_values
from ecoscope.platform.tasks.transformation._sorting import sort_values

PREVIEW_SAMPLE_ROWS = 100
PREVIEW_MAX_VALUES = 50

# The tasks a step may run, by name. Each takes the table as `df` and returns a table;
# to allow another such task, add it here.
STEP_TASKS: dict[str, Callable[..., Any]] = {
    task.__name__: task
    for task in (
        filter_df,
        sort_values,
        map_columns,
        map_values,
        fill_na,
        convert_values_to_timezone,
        add_temporal_index,
        summarize_df,
    )
}


# The keyword of the `steps` form schema that holds `processing_step_schemas()`.
STEP_TASKS_KEYWORD = "ecoscope:step_tasks"


class ProcessingStep(BaseModel):
    """One call of a task in `STEP_TASKS` on the table, with the task's own parameters."""

    model_config = ConfigDict(extra="forbid")

    task: str
    params: dict[str, Any] = {}


@cache
def _params_model(task: str) -> type[BaseModel]:
    """A model of the task's parameters other than `df`, typed as its signature types them."""
    func = STEP_TASKS[task]
    hints = get_type_hints(func, include_extras=True)
    parameters = {name: p for name, p in inspect.signature(func).parameters.items() if name != "df"}
    # Built as a class body would be, not with `create_model`: a parameter with no
    # Python default is left unassigned, so that a default given in its `Field`, as
    # `map_values` gives `missing_values`, stands whatever the pydantic version.
    namespace: dict[str, Any] = {
        "__module__": __name__,
        "__annotations__": {name: hints[name] for name in parameters},
        "model_config": ConfigDict(extra="forbid", arbitrary_types_allowed=True),
        **{name: p.default for name, p in parameters.items() if p.default is not inspect.Parameter.empty},
    }
    return type(task, (BaseModel,), namespace)


def _inlined(node: Any, defs: dict[str, Any], seen: tuple[str, ...] = ()) -> Any:
    """The schema node with each local `$ref` replaced by the definition it names."""
    if isinstance(node, list):
        return [_inlined(item, defs, seen) for item in node]
    if not isinstance(node, dict):
        return node
    ref = node.get("$ref")
    if isinstance(ref, str) and ref.startswith("#/$defs/"):
        name = ref.removeprefix("#/$defs/")
        if name in seen:
            raise ValueError(f"recursive schema definition {name!r}")
        rest = {key: value for key, value in node.items() if key != "$ref"}
        return {**_inlined(defs[name], defs, (*seen, name)), **_inlined(rest, defs, seen)}
    inlined = {key: _inlined(value, defs, seen) for key, value in node.items() if key != "$defs"}
    # Pydantic wraps a reference that has siblings in a one-item `allOf`.
    match inlined.get("allOf"):
        case [dict() as only]:
            del inlined["allOf"]
            inlined = {**only, **inlined}
    return inlined


def processing_step_schemas() -> dict[str, dict[str, Any]]:
    """The JSON schema of the `params` of each task a step may run, by task name.

    It is the schema a step's `params` is validated against: the task's parameters
    as its signature types them, without the table (`df`). A steps editor can render
    a step's form from it. Unlike a workflow's form, it keeps the parameters a task
    hides from that form, such as the `groupers` of `add_temporal_index`. Each schema
    stands alone: its definitions are inlined, so it has no `$ref` or `$defs`.

    The form schema of `apply_processing_steps` carries this map on its `steps`
    property, under `STEP_TASKS_KEYWORD`, which is how it reaches an editor.
    """
    schemas = {task: _params_model(task).model_json_schema() for task in STEP_TASKS}
    return {task: _inlined(schema, schema.get("$defs", {})) for task, schema in schemas.items()}


def _parse(raw: Any) -> tuple[str, dict[str, Any]]:
    """The task name and validated keyword arguments of one item of `steps`."""
    if isinstance(raw, ProcessingStep):
        raw = raw.model_dump()
    if not isinstance(raw, dict):
        raise ValueError(f"a step is an object, not {raw!r}")
    allowed = ", ".join(STEP_TASKS)
    if "task" not in raw:
        raise ValueError(f"a step needs a task, one of: {allowed}")
    task = raw["task"]
    if not isinstance(task, str) or task not in STEP_TASKS:
        raise ValueError(f"unknown task {task!r}; a step runs one of: {allowed}")
    if extra := sorted(set(raw) - {"task", "params"}):
        raise ValueError(f"{task}: a step has a task and its params, not {extra[0]!r}")
    params = raw.get("params") or {}
    if not isinstance(params, dict):
        raise ValueError(f"{task}: params is an object, not {params!r}")
    try:
        validated = _params_model(task).model_validate(params)
    except ValidationError as e:
        error = e.errors(include_url=False)[0]
        where = ".".join(str(part) for part in error["loc"])
        match error["type"]:
            case "missing":
                problem = f"{where} is required"
            case "extra_forbidden":
                problem = f"no parameter {where!r}"
            case _:
                problem = f"{where}: {error['msg']}"
        raise ValueError(f"{task}: {problem}") from None
    return task, {name: getattr(validated, name) for name in type(validated).model_fields}


def _run(df: pd.DataFrame, task: str, kwargs: dict[str, Any]) -> pd.DataFrame:
    try:
        # A shallow copy, so that a task assigning a column leaves the caller's frame alone.
        return STEP_TASKS[task](df=df.copy(deep=False), **kwargs)
    except KeyError as e:
        raise ValueError(f"{task}: column {e.args[0]!r} not found") from e
    except Exception as e:
        raise ValueError(f"{task}: {e}") from e


def _flat(df: pd.DataFrame) -> pd.DataFrame:
    """The frame with its named index levels as leading columns: a table's file has no index."""
    levels = [name for name in df.index.names if name is not None and name not in df.columns]
    return df.reset_index(level=levels) if levels else df


def _skipped_to_empty(df: Any) -> Any:
    return pd.DataFrame() if isinstance(df, SkipSentinel) else df


def _json_cell(value: Any) -> Any:
    if value is None or isinstance(value, (list, tuple, dict, set, np.ndarray, bytes)):
        return None
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if pd.isna(value):
        return None
    if isinstance(value, pd.Timestamp):
        return (value.tz_convert("UTC") if value.tzinfo is not None else value).isoformat()
    return str(value)


def _snapshot(df: pd.DataFrame) -> dict[str, Any]:
    columns = []
    kinds = {}
    for position, name in enumerate(df.columns):
        series = df.iloc[:, position]
        kind = kinds[position] = column_kind(series)
        column: dict[str, Any] = {"name": str(name), "type": kind}
        if kind in ("text", "boolean"):
            distinct = series.dropna().unique()
            if len(distinct) <= PREVIEW_MAX_VALUES:
                column["values"] = sorted(
                    str(bool(v)).lower() if kind == "boolean" else str(v) for v in distinct.tolist()
                )
        columns.append(column)

    head = df.iloc[:PREVIEW_SAMPLE_ROWS]
    cells = [
        [None] * len(head)
        if kinds[position] == "geometry"
        else [_json_cell(v) for v in head.iloc[:, position].astype(object)]
        for position in range(len(df.columns))
    ]
    names = [c["name"] for c in columns]
    return {"rows": len(df), "columns": columns, "sample": [dict(zip(names, row)) for row in zip(*cells)]}


# Tagged `no-cache`: the preview file is written into this run's results folder. A hit
# would return the frame without writing it, and the run would have no preview.
@register(tags=["no-cache"])
def apply_processing_steps(
    df: Annotated[
        AnyDataFrameOrEmpty | AnyGeoDataFrameOrEmpty,
        SkippedDependencyFallback(_skipped_to_empty),
        Field(description="The table's rows", exclude=True),
    ],
    table_id: Annotated[
        str,
        Field(description="The table's id in its data set; names the preview file", exclude=True),
    ],
    root_path: Annotated[str, Field(exclude=True)],
    steps: Annotated[
        list[Any],
        # The form schema is a bare list of objects, with no `oneOf`/`anyOf`: this
        # section is written by the steps editor, not rendered as a form, and a union
        # in a form schema needs a branch selection nothing would make. Each item is
        # validated as a `ProcessingStep`, and its params against its task, when the
        # task runs. The tasks' params schemas ride along under a keyword no form
        # validator reads, for the steps editor to render a step's form from.
        WithJsonSchema({"type": "array", "items": {"type": "object"}, STEP_TASKS_KEYWORD: processing_step_schemas()}),
        Field(
            title="Processing Steps",
            description="Task calls applied to the table in order, each `{task, params}`.",
        ),
    ] = [],  # noqa: B006
) -> Annotated[AnyDataFrameOrEmpty | AnyGeoDataFrameOrEmpty, Field()]:
    """Run a data set table through a list of task calls, and write a preview of each result.

    A step is `{"task": <name>, "params": {...}}`. It calls the registered task of that
    name with the table as `df` and `params` as its other arguments, and the table it
    returns goes to the next step. `params` is validated against the task's own
    signature, so it takes what that task takes in a workflow spec. The tasks a step
    may run are those in `STEP_TASKS`, and `processing_step_schemas()` gives the schema
    of each one's `params`:

    - `filter_df`: keep the rows whose value in a column passes a test.
    - `sort_values`: order the rows by a column.
    - `map_columns`: drop, keep, reorder and rename columns.
    - `map_values`: replace a column's values through a mapping.
    - `fill_na`: fill missing values.
    - `convert_values_to_timezone`: move date columns to a time zone.
    - `add_temporal_index`: add a column per time grouper, holding a date column
      formatted its way and named `TemporalGrouper_<directive>`, for a later
      `summarize_df` to group by.
    - `summarize_df`: replace the rows with one row per group, holding its statistics.
      A statistic cannot be taken over a column the rows are grouped by.

    For example, the number of events per month and type::

        [
            {"task": "filter_df", "params": {"column_name": "event_type", "op": "is_not_empty"}},
            {"task": "add_temporal_index",
             "params": {"time_col": "time", "groupers": [{"temporal_index": "%Y-%m"}]}},
            {"task": "summarize_df",
             "params": {"groupby_cols": ["TemporalGrouper_%Y-%m", "event_type"],
                        "summary_params": [
                            {"display_name": "events", "aggregator": "count", "column": "time"}]}},
        ]

    The preview goes to `data_sets/preview/<table_id>.json` in the results folder: the
    source table first, then the table after each step, each as its row count, its
    columns and their kinds, and its first rows. A step that
    cannot run is recorded there with its error before the task raises, and so is one
    that is not a valid step: an unknown task, or params its task does not accept.

    A table has no index: the named index levels a task leaves on it, as
    `add_temporal_index` and a grouped `summarize_df` do, become its leading columns
    before the next step, the preview and the return.

    A table with no rows is not passed to the tasks: each step is still validated, and
    the table goes through unchanged.

    Args:
        df: The table's rows. A skipped upstream step is an empty table.
        table_id: The table's id in its data set.
        root_path: The results folder.
        steps: The steps, applied in order, each a `ProcessingStep` or the JSON object
            of one.

    Returns:
        The table after the last step, of whatever shape the steps gave it: still a
        GeoDataFrame after steps that keep the geometry, a plain DataFrame once
        summarized.
    """
    table_filename(table_id)
    filename = f"data_sets/preview/{table_id}.json"
    preview_path, _ = _get_path(root_path, filename)

    frame: pd.DataFrame = df
    entries = [_snapshot(frame)]
    failure = None
    for number, raw in enumerate(steps, start=1):
        try:
            task, kwargs = _parse(raw)
            if len(frame):
                frame = _flat(_run(frame, task, kwargs))
        except Exception as e:
            failure = ValueError(f"step {number}: {e}")
            entries.append({"rows": None, "columns": [], "sample": [], "error": str(failure)})
            break
        entries.append({**_snapshot(frame), "error": None})

    if isinstance(preview_path, Path):
        preview_path.parent.mkdir(parents=True, exist_ok=True)
    _persist_text(json.dumps({"table_id": table_id, "steps": entries}, allow_nan=False), root_path, filename)
    if failure is not None:
        raise failure
    return frame  # type: ignore[return-value]
