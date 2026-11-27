import logging
from enum import Enum
from typing import Annotated, Any, Literal, cast

import numpy as np
import pandas as pd
from pydantic import Field
from pydantic.json_schema import SkipJsonSchema, WithJsonSchema
from wt_registry import register

from ecoscope.platform.annotations import AnyDataFrame

logger = logging.getLogger(__name__)

ColumnKind = Literal["text", "number", "date", "boolean", "geometry", "other"]

# An operand may be given as text (as a form sends it) or as a number or boolean (as a
# spec or a data set's processing step can). Either way it is read as the column's type.
# Its schema stays a plain string, so forms and the steps editor show a text box.
Operand = str | int | float | bool


class ComparisonOperator(Enum):
    EQUAL = "equal"
    GE = "ge"
    GT = "gt"
    LE = "le"
    LT = "lt"
    NE = "ne"
    IS_ANY_OF = "is_any_of"
    IS_NONE_OF = "is_none_of"
    IS_EMPTY = "is_empty"
    IS_NOT_EMPTY = "is_not_empty"

    @classmethod
    def from_string(cls, value: str):
        try:
            return cls(value.lower())
        except ValueError:
            raise ValueError(f"'{value}' is not a valid FieldType. Valid options are: {[e.value for e in cls]}")


def column_kind(series: pd.Series) -> ColumnKind:
    """What a column holds, as a filter operand is read and a preview describes it."""
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
    if all(issubclass(t, (int, float, np.integer, np.floating)) and not issubclass(t, bool) for t in types):
        return "number"
    return "other"


def _coerce(value: Any, kind: ColumnKind, series: pd.Series, column: str) -> Any:
    problem = ValueError(f"cannot compare value {value!r} with {kind} column {column!r}")
    if kind == "number":
        if isinstance(value, bool):
            raise problem
        try:
            return float(value)
        except (ValueError, TypeError):
            raise problem from None
    if kind == "boolean":
        if isinstance(value, (bool, np.bool_)):
            return bool(value)
        if isinstance(value, str) and value.strip().lower() in ("true", "false"):
            return value.strip().lower() == "true"
        raise problem
    if kind == "date":
        if not isinstance(value, (str, pd.Timestamp)):
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
        return timestamp.tz_localize(column_tz) if timestamp.tzinfo is None else timestamp
    return str(value)


@register()
def filter_df(
    df: Annotated[
        AnyDataFrame,
        Field(
            description="The dataframe.",
            exclude=True,
        ),
    ],
    column_name: Annotated[str, Field(description="The column name to filter on.")],
    op: Annotated[ComparisonOperator, Field(description="The comparison operator")],
    value: Annotated[
        Operand,
        WithJsonSchema({"type": "string"}),
        Field(description="The comparison operand, read as the column's type. Used by equal, ne, gt, ge, lt and le."),
    ] = "",
    reset_index: Annotated[
        bool,
        Field(description="Number the kept rows 0, 1, 2, ... instead of keeping their index. Default is False."),
    ] = False,
    values: Annotated[
        list[Annotated[Operand, WithJsonSchema({"type": "string"})]] | SkipJsonSchema[None],
        Field(description="The operands of is_any_of and is_none_of, each read as the column's type."),
    ] = None,
) -> AnyDataFrame:
    """Keep the rows whose value in one column passes a test.

    An operand is read as the column's type: a number for a numeric column, `true` or
    `false` for a boolean one, a timestamp for a date one (one without an offset is
    taken in the column's own time zone), and text otherwise. An operand that cannot
    be read that way is an error. An operand may be given as text or, from a spec or
    a processing step, as a number or boolean: `30` and `"30"` filter the same way.

    A row with no value passes only `ne` and `is_none_of`. `is_empty` keeps the rows
    with no value, and in a text column those with an empty string; `is_not_empty`
    keeps the rest. `is_any_of` with no `values` keeps nothing, and `is_none_of` with
    none keeps everything. A geometry column can only be tested for emptiness.

    `reset_index` renumbers the kept rows from 0 and drops the old index; it does not
    keep it as a column.
    """
    series = df[column_name]
    kind = column_kind(series)

    if op in (ComparisonOperator.IS_EMPTY, ComparisonOperator.IS_NOT_EMPTY):
        present = series.notna()
        if kind == "text":
            present &= series.astype(object) != ""
        result_df = df[present if op == ComparisonOperator.IS_NOT_EMPTY else ~present]
    else:
        if kind == "geometry":
            raise ValueError(f"geometry column {column_name!r} can only be tested with is_empty or is_not_empty")
        known = series[series.notna()]
        if kind in ("text", "other"):
            known = known.astype(object).map(str)
        elif kind == "boolean":
            known = known.astype(bool)
        elif kind == "number" and not pd.api.types.is_numeric_dtype(known.dtype):
            known = pd.to_numeric(known.astype(object))

        if op in (ComparisonOperator.IS_ANY_OF, ComparisonOperator.IS_NONE_OF):
            operands = [_coerce(v, kind, series, column_name) for v in values or []]
            passed = known.isin(operands)
            if op == ComparisonOperator.IS_NONE_OF:
                passed = ~passed
        else:
            operand = _coerce(value, kind, series, column_name)
            match op:
                case ComparisonOperator.EQUAL:
                    passed = known == operand
                case ComparisonOperator.GE:
                    passed = known >= operand
                case ComparisonOperator.GT:
                    passed = known > operand
                case ComparisonOperator.LE:
                    passed = known <= operand
                case ComparisonOperator.LT:
                    passed = known < operand
                case ComparisonOperator.NE:
                    passed = known != operand
        # A row with no value passes only the negative tests. The mask is set by
        # position, since the index may repeat a label.
        mask = np.full(len(series), op in (ComparisonOperator.NE, ComparisonOperator.IS_NONE_OF))
        mask[series.notna().to_numpy()] = passed.to_numpy(dtype=bool)
        result_df = df[mask]

    if reset_index:
        result_df = result_df.reset_index(drop=True)

    return cast(AnyDataFrame, result_df)
