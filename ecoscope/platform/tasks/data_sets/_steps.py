import json
import math
from pathlib import Path
from typing import Annotated, Any, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError, WithJsonSchema, model_validator
from wt_registry import register
from wt_task.skip import SkippedDependencyFallback, SkipSentinel

from ecoscope.platform.annotations import AnyDataFrameOrEmpty, AnyGeoDataFrameOrEmpty, CacheableModel
from ecoscope.platform.data_sets import table_filename
from ecoscope.platform.serde import _get_path, _persist_text

PREVIEW_SAMPLE_ROWS = 100
PREVIEW_MAX_VALUES = 50

FilterOp = Literal["is_any_of", "is_none_of", "eq", "ne", "gt", "ge", "lt", "le", "is_empty", "is_not_empty"]
MeasureOp = Literal["count", "sum", "mean", "min", "max", "median", "nunique"]
TimeInterval = Literal["day", "week", "month", "year"]
ColumnKind = Literal["text", "number", "date", "boolean", "geometry", "other"]

_SINGLE_VALUE_OPS = ("eq", "ne", "gt", "ge", "lt", "le")
_NUMERIC_ONLY_OPS = ("sum", "mean", "median")


class FilterStep(BaseModel, CacheableModel):
    """Keep the rows whose value in one column passes a test."""

    model_config = ConfigDict(title="Filter")

    type: Literal["filter"]
    column: Annotated[str, Field(title="Column", description="The column to test.")]
    op: Annotated[FilterOp, Field(title="Condition")]
    values: Annotated[
        list[bool | int | float | str],
        Field(
            title="Values",
            description=(
                "What to compare with. The comparisons (eq, ne, gt, ge, lt, le) use the first value; "
                "is_empty and is_not_empty use none."
            ),
        ),
    ] = []


class TimeBucket(BaseModel, CacheableModel):
    """Group rows by the day, week, month or year a date column falls in."""

    model_config = ConfigDict(title="Time Bucket")

    column: Annotated[str, Field(title="Column", description="The date column to bucket.")]
    interval: Annotated[TimeInterval, Field(title="Interval")]


class Measure(BaseModel, CacheableModel):
    """One value computed per group."""

    model_config = ConfigDict(title="Measure")

    op: Annotated[MeasureOp, Field(title="Statistic")]
    column: Annotated[
        str | None,
        Field(title="Column", description="The column to compute over. Not used by count."),
    ] = None

    @model_validator(mode="after")
    def _column_unless_count(self) -> "Measure":
        if self.op != "count" and not self.column:
            raise ValueError(f"measure {self.op!r} needs a column")
        return self

    @property
    def name(self) -> str:
        return "count" if self.op == "count" else f"{self.op}_{self.column}"


class AggregateStep(BaseModel, CacheableModel):
    """Replace the rows with one row per group, holding the group's measures."""

    model_config = ConfigDict(title="Aggregate")

    type: Literal["aggregate"]
    group_by: Annotated[
        list[str],
        Field(title="Group By", description="Columns whose distinct values make the groups."),
    ] = []
    time_bucket: Annotated[TimeBucket | None, Field(title="Time Bucket")] = None
    measures: Annotated[
        list[Measure],
        Field(title="Measures", description="Values computed per group. Counts rows when none is given."),
    ] = []

    @model_validator(mode="after")
    def _something_to_do(self) -> "AggregateStep":
        if not (self.group_by or self.time_bucket or self.measures):
            raise ValueError("an aggregate step needs at least one of group_by, time_bucket and measures")
        if not self.measures:
            self.measures = [Measure(op="count")]
        return self


ProcessingStep = Annotated[FilterStep | AggregateStep, Field(discriminator="type")]


_STEP: TypeAdapter[FilterStep | AggregateStep] = TypeAdapter(ProcessingStep)


def _parse(raw: Any, number: int) -> FilterStep | AggregateStep:
    """The step model of one item of `steps`, or a `ValueError` saying what is wrong with it."""
    if isinstance(raw, (FilterStep, AggregateStep)):
        return raw
    if not isinstance(raw, dict):
        raise ValueError(f"step {number}: a step is an object, not {raw!r}")
    try:
        return _STEP.validate_python(raw)
    except ValidationError as e:
        error = e.errors(include_url=False)[0]
    where = ".".join(str(part) for part in error["loc"] if part not in ("filter", "aggregate"))
    match error["type"]:
        case "union_tag_not_found":
            problem = "a step needs a type, 'filter' or 'aggregate'"
        case "union_tag_invalid":
            problem = f"unknown step type {raw.get('type')!r}; a step is a 'filter' or an 'aggregate'"
        case "value_error":
            problem = str(error["ctx"]["error"])
        case "missing":
            problem = f"{where} is required"
        case _:
            problem = f"{where}: {error['msg']}"
    raise ValueError(f"step {number}: {problem}")


def _skipped_to_empty(df: Any) -> Any:
    return pd.DataFrame() if isinstance(df, SkipSentinel) else df


def _column_kind(series: pd.Series) -> ColumnKind:
    dtype = series.dtype
    if dtype.name == "geometry":
        return "geometry"
    if pd.api.types.is_bool_dtype(dtype):
        return "boolean"
    if pd.api.types.is_numeric_dtype(dtype):
        return "number"
    if pd.api.types.is_datetime64_any_dtype(dtype):
        return "date"
    if isinstance(dtype, pd.CategoricalDtype):
        series = series.astype(object)
    elif not pd.api.types.is_object_dtype(dtype):
        return "text" if pd.api.types.is_string_dtype(dtype) else "other"
    types = {type(v) for v in series[series.notna()]}
    if types <= {str}:
        return "text"
    if all(issubclass(t, (bool, np.bool_)) for t in types):
        return "boolean"
    return "other"


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
        kind = kinds[position] = _column_kind(series)
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


def _require_columns(df: pd.DataFrame, columns: list[str], done: int) -> None:
    for column in columns:
        if column not in df.columns:
            where = f"after step {done}" if done else "in the source table"
            raise ValueError(f"column {column!r} not found {where}")


def _coerce(value: Any, kind: ColumnKind, series: pd.Series, column: str) -> Any:
    problem = ValueError(f"cannot compare value {value!r} with {kind} column {column!r}")
    if kind == "number":
        if isinstance(value, bool):
            raise problem
        try:
            return float(value)
        except ValueError:
            raise problem from None
    if kind == "boolean":
        if isinstance(value, bool):
            return value
        if isinstance(value, str) and value.strip().lower() in ("true", "false"):
            return value.strip().lower() == "true"
        raise problem
    if kind == "date":
        if not isinstance(value, str):
            raise problem
        try:
            timestamp = pd.Timestamp(value)
        except (ValueError, TypeError):
            raise problem from None
        if pd.isna(timestamp):
            raise problem
        column_tz = getattr(series.dtype, "tz", None)
        if column_tz is None:
            return timestamp.tz_convert("UTC").tz_localize(None) if timestamp.tzinfo is not None else timestamp
        return timestamp.tz_localize("UTC") if timestamp.tzinfo is None else timestamp
    return str(value)


def _filter(df: pd.DataFrame, step: FilterStep, done: int) -> pd.DataFrame:
    if len(df.columns) == 0:
        return df
    _require_columns(df, [step.column], done)
    series = df[step.column]
    kind = _column_kind(series)
    present = series.notna()
    if kind == "text":
        present &= series.astype(object) != ""

    if step.op == "is_empty":
        return df[~present]
    if step.op == "is_not_empty":
        return df[present]

    if kind == "geometry":
        raise ValueError(f"geometry column {step.column!r} can only be tested with is_empty or is_not_empty")
    if step.op in _SINGLE_VALUE_OPS and not step.values:
        raise ValueError(f"filter {step.op!r} on column {step.column!r} needs a value")
    values = [_coerce(v, kind, series, step.column) for v in step.values]

    known = series[series.notna()]
    if kind in ("text", "other"):
        known = known.astype(object).map(str)
    elif kind == "boolean":
        known = known.astype(bool)
    match step.op:
        case "is_any_of":
            passed = known.isin(values)
        case "is_none_of":
            passed = ~known.isin(values)
        case "eq":
            passed = known == values[0]
        case "ne":
            passed = known != values[0]
        case "gt":
            passed = known > values[0]
        case "ge":
            passed = known >= values[0]
        case "lt":
            passed = known < values[0]
        case "le":
            passed = known <= values[0]
    # A row with no value passes only the negative tests.
    mask = passed.reindex(series.index, fill_value=step.op in ("is_none_of", "ne"))
    return df[mask.astype(bool)]


def _bucket_start(series: pd.Series, interval: TimeInterval) -> pd.Series:
    utc = series.dt.tz_localize("UTC") if series.dt.tz is None else series.dt.tz_convert("UTC")
    day = utc.dt.floor("D")
    match interval:
        case "day":
            return day
        case "week":
            return day - pd.to_timedelta(day.dt.dayofweek, unit="D")
        case "month":
            return day - pd.to_timedelta(day.dt.day - 1, unit="D")
        case "year":
            return day - pd.to_timedelta(day.dt.dayofyear - 1, unit="D")


def _aggregate(df: pd.DataFrame, step: AggregateStep, done: int) -> pd.DataFrame:
    bucket = step.time_bucket
    keys = ([bucket.column] if bucket else []) + step.group_by
    names = keys + [m.name for m in step.measures]
    for name in names:
        if names.count(name) > 1:
            raise ValueError(f"the aggregate would produce column {name!r} twice")

    if len(df.columns) == 0:
        empty: dict[str, pd.Series] = {name: pd.Series(dtype=object) for name in names}
        if bucket:
            empty[bucket.column] = pd.Series(dtype="datetime64[ns, UTC]")
        for measure in step.measures:
            empty[measure.name] = pd.Series(dtype="int64" if measure.op in ("count", "nunique") else "float64")
        return pd.DataFrame(empty)

    measured = [m.column for m in step.measures if m.column is not None]
    _require_columns(df, keys + measured, done)
    kinds = {column: _column_kind(df[column]) for column in dict.fromkeys(keys + measured)}
    for column in step.group_by:
        if kinds[column] == "geometry":
            raise ValueError(f"cannot group by geometry column {column!r}")
    if bucket and kinds[bucket.column] != "date":
        raise ValueError(f"time bucket column {bucket.column!r} is not a date column")
    for measure in step.measures:
        if measure.op in _NUMERIC_ONLY_OPS and kinds[measure.column] != "number":  # type: ignore[index]
            raise ValueError(f"cannot {measure.op} {kinds[measure.column]} column {measure.column!r}")  # type: ignore[index]

    # The row counter is positional so that it cannot collide with a column's name.
    work = pd.DataFrame({i: df[column] for i, column in enumerate(dict.fromkeys(keys + measured))})
    position = {column: i for i, column in enumerate(dict.fromkeys(keys + measured))}
    rows = len(position)
    work[rows] = 1
    if bucket:
        work[position[bucket.column]] = _bucket_start(work[position[bucket.column]], bucket.interval)
    specs = {
        str(i): (rows, "size") if m.op == "count" else (position[m.column], m.op)  # type: ignore[index]
        for i, m in enumerate(step.measures)
    }
    if len(work) == 0:
        result = work.iloc[:, : len(keys)].copy()
        for i, measure in enumerate(step.measures):
            result[f"m{i}"] = pd.Series(dtype="int64" if measure.op in ("count", "nunique") else "float64")
    elif keys:
        grouped = work.groupby([position[k] for k in keys], dropna=False, sort=True, observed=True)
        result = grouped.agg(**specs).reset_index()
    else:
        result = pd.DataFrame({name: [work[column].agg(op)] for name, (column, op) in specs.items()})
    result.columns = pd.Index(names)
    return result


def _apply(df: pd.DataFrame, step: FilterStep | AggregateStep, done: int) -> pd.DataFrame:
    if isinstance(step, FilterStep):
        return _filter(df, step, done)
    return _aggregate(df, step, done)


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
        # in a form schema needs a branch selection nothing would make. The items are
        # validated as `ProcessingStep` when the task runs.
        WithJsonSchema({"type": "array", "items": {"type": "object"}}),
        Field(title="Processing Steps", description="Filter and aggregate steps, applied in order."),
    ] = [],  # noqa: B006
) -> Annotated[AnyDataFrameOrEmpty | AnyGeoDataFrameOrEmpty, Field()]:
    """Apply a data set table's filter and aggregate steps, and write a preview of each result.

    The preview goes to `data_sets/preview/<table_id>.json` in the results folder: the
    source table first, then the table after each step, each as its row count, its
    columns and their kinds, and its first rows. A step that cannot run is recorded
    there with its error before the task raises, and so is one that is not a valid
    step.

    Args:
        df: The table's rows. A skipped upstream step is an empty table.
        table_id: The table's id in its data set.
        root_path: The results folder.
        steps: The steps, applied in order, each a `ProcessingStep` or the JSON object
            of one. A filter keeps matching rows; an aggregate replaces the rows with
            one per group and drops the geometry.

    Returns:
        The table after the last step: a plain DataFrame once aggregated.
    """
    table_filename(table_id)
    filename = f"data_sets/preview/{table_id}.json"
    preview_path, _ = _get_path(root_path, filename)

    frame: pd.DataFrame = df
    entries = [_snapshot(frame)]
    failure = None
    for done, raw in enumerate(steps):
        try:
            frame = _apply(frame, _parse(raw, done + 1), done)
        except Exception as e:
            failure = e
            entries.append({"rows": None, "columns": [], "sample": [], "error": str(e)})
            break
        entries.append({**_snapshot(frame), "error": None})

    if isinstance(preview_path, Path):
        preview_path.parent.mkdir(parents=True, exist_ok=True)
    _persist_text(json.dumps({"table_id": table_id, "steps": entries}, allow_nan=False), root_path, filename)
    if failure is not None:
        raise ValueError(str(failure)) from failure
    return frame  # type: ignore[return-value]
