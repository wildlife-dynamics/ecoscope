"""How an `ecoscope.platform` task return becomes bytes, and back again.

Only leaves live here. Which codec applies is declared by the task's return
type -- `CodecRef` on a type alias, `__wt_codec__` on a class -- and wt-task
assembles it into one `wt_task.cache.Tree` per step when the step first runs.
Nothing in this module inspects a value to decide what it is.

The digest half is also what `persist_df` and friends use for their
content-addressed filenames, so there is one content-identity implementation
rather than one per caller.

Deliberately a module, not a package: `[tool.setuptools].packages` in
`pyproject.toml` is an explicit list, so a new subpackage would silently not
ship.
"""

from __future__ import annotations

import base64
import datetime
import decimal
import hashlib
import io
import json
import struct
import uuid
import zoneinfo
from typing import TYPE_CHECKING, Any

import pandas as pd
import pyarrow as pa  # type: ignore[import-untyped]
import pytz
from wt_task.cache import SKIP, Codec, Tree, Undigestible
from wt_task.cache._tree import strict  # the active run's setting; not exported

if TYPE_CHECKING:
    from pydantic import BaseModel

    from ecoscope.platform.annotations import AnyDataFrame

FRAME_CODEC_ID = "ecoscope.frame/v3"
ARROW_CODEC_ID = "ecoscope.arrow/v1"
KEYED_CODEC_ID = "ecoscope.keyed/v1"
GROUPER_CODEC_ID = "ecoscope.grouper/v1"
MODEL_CODEC_ID = "ecoscope.model/v1"


class UnencodableValue(Undigestible):
    """Raised by a leaf handed a value it cannot carry losslessly."""


# --------------------------------------------------------------------------- #
# dataframes
# --------------------------------------------------------------------------- #

# Schema-metadata key naming the object columns a Studio data set stores as JSON
# trees (`data_sets.frame_to_table`). Arrow types such a column structurally,
# which loses information in two directions: a `list` and a `tuple` cell both
# come back as `ndarray`, and a column of dicts becomes a struct whose schema is
# the *union* of the keys seen, so `{"k": 1}` next to `{"k": 2, "j": "x"}` reads
# back as `{"k": 1, "j": None}`.
TREE_COLUMNS_KEY = b"ecoscope:tree_columns"

# Schema-metadata key holding what `encode_dataframe` changed to fit a frame into
# parquet, for `decode_dataframe` to undo. Written only when non-empty, so a
# frame parquet carries as-is pays no metadata rewrite.
FRAME_SPEC_KEY = b"ecoscope:frame"


def has_geometry(df: AnyDataFrame) -> bool:
    """Whether `df` carries at least one geometry-dtype column."""
    import geopandas as gpd  # type: ignore[import-untyped]

    return any(isinstance(dtype, gpd.array.GeometryDtype) for dtype in df.dtypes)


def _fallback_hash(obj: Any) -> Any:
    import pandas as pd

    try:
        return pd.util.hash_pandas_object(obj, index=False).values
    except (TypeError, ValueError, AttributeError):
        # Cells pandas can't hash (lists/dicts in event details). `repr`, not
        # `str`, which collapses 1 and "1".
        try:
            return pd.util.hash_pandas_object(obj.map(repr), index=False).values
        except Exception as e:
            # Fail loudly if we hit a value whose own `__repr__` fails.
            label = obj.name if isinstance(obj, pd.Series) else "<index>"
            raise TypeError(f"cannot content-hash column {label!r}: values are neither hashable nor reprable") from e


def _frame_metadata(df: AnyDataFrame) -> dict[str, Any]:
    """The parts of a frame's identity that live outside its cells.

    `hash_pandas_object` sees values only, so two frames differing just in
    projection, in `attrs`, or in what their index levels are called hash
    equal without this -- which, for `persist_df`'s skip-if-exists, means
    serving the wrong file.
    """
    import geopandas as gpd  # type: ignore[import-untyped]

    crs: dict[str, str | None] = {}
    for position, (name, dtype) in enumerate(zip(df.columns, df.dtypes)):
        if isinstance(dtype, gpd.array.GeometryDtype):
            # From the array: a geometry column of a plain DataFrame is a Series, without `.crs`.
            column_crs = df.iloc[:, position].array.crs
            crs[str(name)] = None if column_crs is None else column_crs.to_string()

    index = df.index
    levels = [index.get_level_values(i) for i in range(index.nlevels)]
    is_geo = isinstance(df, gpd.GeoDataFrame)
    return {
        "geo": is_geo,
        "active_geometry": df._geometry_column_name if is_geo else None,
        # Values and dtype alone can't tell a `RangeIndex` from an int64 `Index`, and tasks branch on it.
        "index_type": type(index).__name__,
        "index_names": [repr(n) for n in index.names],
        "index_dtypes": [_dtype_token(level.dtype) for level in levels],
        "index_freq": index.freqstr if isinstance(index, (pd.DatetimeIndex, pd.TimedeltaIndex)) else None,
        "index_missing": [_missing_kinds(level) for level in levels],
        "columns_type": type(df.columns).__name__,
        "columns_names": [repr(n) for n in df.columns.names],
        "missing": {str(i): kinds for i in range(df.shape[1]) if (kinds := _missing_kinds(df.iloc[:, i]))},
        "crs": crs,
        "attrs": {str(k): v for k, v in df.attrs.items()},
        "flags": df.flags.allows_duplicate_labels,
    }


def _dtype_token(dtype: Any) -> str:
    """`str(dtype)`, plus what it omits: a categorical's categories and order, a zone's library, a string's storage."""
    if isinstance(dtype, pd.CategoricalDtype) and dtype.categories is not None:
        return f"category/{dtype.ordered}/{_dtype_token(dtype.categories.dtype)}/{list(dtype.categories)!r}"
    if (tz := getattr(dtype, "tz", None)) is not None:
        return f"{dtype}/{type(tz).__module__}"
    if isinstance(dtype, pd.StringDtype):
        return repr(dtype)
    return str(dtype)


def _missing_kind(value: Any) -> str | None:
    if value is None:
        return "none"
    if value is pd.NA:
        return "na"
    if isinstance(value, float) and value != value:
        return "nan"
    return None


def _missing_kinds(values: Any) -> list[str]:
    """Which of `None`, NaN and `pd.NA` an object column holds: pandas hashes them alike, parquet stores all as null."""
    import numpy as np

    if values.dtype != object:
        return []
    array = np.asarray(values, dtype=object)
    return sorted({kind for v in array[np.asarray(pd.isna(array))] if (kind := _missing_kind(v))})


def _cell_types(column: pd.Series) -> set[type]:
    """Types of a column's cells, leaving out pandas' missing values (`None`, float NaN)."""
    return {type(v) for v in column if not (v is None or (isinstance(v, float) and v != v))}


_TEMPORAL_CELL_TYPES = (datetime.datetime, datetime.timedelta, type(pd.NaT))


_REPR_EXACT = frozenset({str, int, float, bool, type(None)})


def _repr_is_exact(value: Any) -> bool:
    """Whether `repr` alone is exact and stable: only these scalars, in plain lists, tuples and dicts."""
    kind = type(value)
    if kind in _REPR_EXACT:
        return True
    if kind is list or kind is tuple:
        return all(_repr_is_exact(v) for v in value)
    if kind is dict:
        return all(type(k) in _REPR_EXACT and _repr_is_exact(v) for k, v in value.items())
    return False


def _cell_token(value: Any) -> str:
    """`repr`, plus what it leaves out: a timestamp's unit, naive fold and zone library, every element of a large
    array, and nothing of a set's iteration order, which varies with `PYTHONHASHSEED`.
    """
    import numpy as np

    if isinstance(value, pd.Timestamp):
        return f"{value!r}/{value.unit}/{value.fold}/{type(value.tz).__module__}"
    if isinstance(value, pd.Timedelta):
        return f"{value!r}/{value.unit}"
    if _repr_is_exact(value):
        return repr(value)
    if isinstance(value, (list, tuple)):
        return f"{type(value).__name__}[{','.join(_cell_token(v) for v in value)}]"
    if isinstance(value, dict):
        return f"{type(value).__name__}{{{','.join(f'{_cell_token(k)}:{_cell_token(v)}' for k, v in value.items())}}}"
    if isinstance(value, (set, frozenset)):
        return f"{type(value).__name__}{{{','.join(sorted(_cell_token(v) for v in value))}}}"
    if isinstance(value, np.ndarray):
        if value.dtype == object:
            return f"ndarray/{value.shape}[{','.join(_cell_token(v) for v in value.ravel())}]"
        content = hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()
        return f"ndarray/{value.dtype.str}/{value.shape}/{content}"
    return repr(value)


def _hashed_by_cell(column: Any) -> bool:
    """Whether pandas' own hash would lose a distinction between this column's cells.

    It hashes an object column through `str`, without raising, so `1` digests
    like `"1"`, a Timestamp like its text whatever its zone, and `None` like NaN.
    """
    if column.dtype != object:
        return False
    types = _cell_types(column)
    return len(types) > 1 or any(t is not str for t in types) or len(_missing_kinds(column)) > 1


def _by_cell_hash(column: Any) -> Any:
    try:
        tokens = [_cell_token(v) for v in column]
    except Exception as e:
        label = column.name if isinstance(column, pd.Series) else "<index>"
        raise TypeError(f"cannot content-hash column {label!r}: values are neither hashable nor reprable") from e
    return pd.util.hash_pandas_object(pd.Series(tokens, dtype=object), index=False).values


def digest_dataframe(df: AnyDataFrame) -> str:
    """Return a full sha256 over a dataframe's content.

    Covers every row and the index, column names and dtypes, index names and
    dtypes, per-column CRS, and `df.attrs`. It identifies the *content*, not
    the bytes on disk: the same frame encoded two ways digests the same, which
    is what lets the cache key stay stable across runs even though parquet
    bytes are not.
    """
    import numpy as np
    import pandas as pd

    digest = hashlib.sha256()
    levels = [df.index.get_level_values(i) for i in range(df.index.nlevels)]
    columns = [df.iloc[:, i] for i in range(len(df.columns))]
    levels_by_cell = [_hashed_by_cell(level) for level in levels]
    by_cell = [_hashed_by_cell(column) for column in columns]
    try:
        if any(by_cell) or any(levels_by_cell):
            raise TypeError("an object column is hashed cell by cell")
        digest.update(np.ascontiguousarray(pd.util.hash_pandas_object(df).values))
    except (TypeError, ValueError, AttributeError):
        # Only now, and only for the columns that genuinely fail, pay per-column
        # costs. Every row still contributes; hashing a `head(5)` sample here
        # let two frames sharing shape and first rows collide.
        if any(levels_by_cell):
            for level, is_by_cell in zip(levels, levels_by_cell):
                digest.update(np.ascontiguousarray(_by_cell_hash(level) if is_by_cell else _fallback_hash(level)))
        else:
            digest.update(np.ascontiguousarray(_fallback_hash(df.index)))
        for column, is_by_cell in zip(columns, by_cell):
            digest.update(np.ascontiguousarray(_by_cell_hash(column) if is_by_cell else _fallback_hash(column)))
    digest.update(json.dumps([[repr(c), _dtype_token(dt)] for c, dt in zip(df.columns, df.dtypes)]).encode())
    digest.update(json.dumps(_frame_metadata(df), sort_keys=True, default=str).encode())
    return digest.hexdigest()


_TREE_CELL_TYPES = (list, tuple, dict, *_TEMPORAL_CELL_TYPES)


def tree_columns(df: AnyDataFrame) -> list[str]:
    """Object columns Arrow cannot carry faithfully.

    Those holding a `list`, `tuple` or `dict` in any cell: one rule for all
    three rather than per-kind repair on the way out, because the structure is
    what Arrow flattens, and reconstructing it from a typed column means
    guessing, which is how a column of genuine lists becomes a column of tuples.
    Those holding datetimes or timedeltas, which Arrow retypes as one column in
    one zone and unit. And those mixing scalar types, which Arrow refuses (`1`
    beside `"1"`) or coerces (`1` beside `1.5` comes back `1.0`).
    """
    found = []
    for position, (name, dtype) in enumerate(zip(df.columns, df.dtypes)):
        if dtype != object:
            continue
        types = _cell_types(df.iloc[:, position])
        if len(types) > 1 or any(issubclass(t, _TREE_CELL_TYPES) for t in types):
            found.append(str(name))
    return found


def _fixed_offset(tz: datetime.timezone) -> tuple[float, str | None]:
    """`(offset seconds, name)`, the name `None` unless one was given: `timezone.utc` is not `timezone(0, "UTC")`."""
    offset, *name = tz.__getinitargs__()  # type: ignore[attr-defined]
    return offset.total_seconds(), name[0] if name else None


def _restore_timezone(seconds: float, name: str | None) -> datetime.timezone:
    offset = datetime.timedelta(seconds=seconds)
    return datetime.timezone(offset) if name is None else datetime.timezone(offset, name)


def _timezone_to_json(tz: datetime.tzinfo | None) -> dict[str, Any] | None:
    """Name a zone so that decode rebuilds the same kind of object.

    A cell's `repr` spells out its zone object and a mixed column is digested by
    `repr`, so the same zone under another implementation is a different digest.
    """
    if tz is None:
        return None
    if isinstance(tz, datetime.timezone):
        return {"k": "offset", "v": list(_fixed_offset(tz))}
    if isinstance(tz, zoneinfo.ZoneInfo) and tz.key is not None:
        return {"k": "zoneinfo", "v": tz.key}
    if isinstance(tz, pytz.BaseTzInfo) and tz.zone is not None:
        return {"k": "pytz", "v": tz.zone}
    if isinstance(tz, pytz._FixedOffset) and (offset := tz.utcoffset(None)) is not None:
        return {"k": "pytz_offset", "v": offset.total_seconds() / 60}
    raise UnencodableValue(f"timezone {tz!r} cannot be stored inside a dataframe cell")


def _timezone_from_json(node: dict[str, Any]) -> datetime.tzinfo:
    match node:
        case {"k": "offset", "v": [seconds, name]}:
            return _restore_timezone(seconds, name)
        case {"k": "zoneinfo", "v": key}:
            return zoneinfo.ZoneInfo(key)
        case {"k": "pytz", "v": zone}:
            return pytz.timezone(zone)
        case {"k": "pytz_offset", "v": minutes}:
            return pytz.FixedOffset(minutes)
        case _:
            raise ValueError(f"unreadable timezone node {node!r}")


def _pytz_variants(wall: datetime.datetime, zone: Any) -> list[datetime.tzinfo]:
    """The tzinfo objects pytz attaches to `wall`: the zone itself under `replace`, then each `is_dst` reading.

    A pytz zone is one object per offset it has ever had, and a cell's `repr`
    names which.
    """
    return [zone, *(zone.localize(wall, is_dst=is_dst).tzinfo for is_dst in (False, True))]


def _cell_to_json(value: Any) -> dict[str, Any]:
    """Describe a structural cell as a tree JSON can carry.

    Structure is spelled out rather than left to `json.dumps` because a `list`
    and a `tuple` cell are not the same value, and nothing else in the frame
    records which one a column held.
    """
    import numpy as np

    # Before the JSON scalars: `np.float64` is a `float`, and would come back as one.
    if isinstance(value, (np.bool_, np.number, np.str_, np.bytes_)):
        return {"k": "np", "d": value.dtype.str, "v": _cell_to_json(value.item())}
    if value is pd.NA:
        return {"k": "na"}
    if value is None or isinstance(value, (str, bool, int, float)):
        return {"k": "j", "v": value}
    if isinstance(value, complex):
        return {"k": "complex", "v": [value.real, value.imag]}
    if isinstance(value, (list, tuple)):
        return {"k": "list" if isinstance(value, list) else "tuple", "v": [_cell_to_json(v) for v in value]}
    if isinstance(value, np.ndarray) and value.dtype.kind in "biufcUS":
        return {
            "k": "ndarray",
            "d": value.dtype.str,
            "s": list(value.shape),
            "v": _cell_to_json(value.ravel().tolist()),
        }
    if isinstance(value, (set, frozenset)):
        return {"k": type(value).__name__, "v": [_cell_to_json(v) for v in value]}
    if isinstance(value, dict):
        if not all(isinstance(k, str) for k in value):
            return {"k": "pairs", "v": [[_cell_to_json(k), _cell_to_json(v)] for k, v in value.items()]}
        return {"k": "dict", "v": {k: _cell_to_json(v) for k, v in value.items()}}
    if value is pd.NaT:  # before the datetime kinds: it is an instance of `datetime`
        return {"k": "nat"}
    # Integers in the cell's own unit: text is exact only for years 1-9999, and
    # nanoseconds overflow for a coarser unit.
    if isinstance(value, pd.Timestamp):
        return {
            "k": "timestamp",
            "v": int(value.asm8.astype("int64")),  # the UTC instant, if aware
            "u": value.unit,
            "f": value.fold if value.tz is None else 0,
            "z": _timezone_to_json(value.tz),
        }
    if isinstance(value, datetime.datetime):
        # The wall clock, not the instant: one that its zone would not produce
        # (a gap time, pytz's LMT) has to come back as it was.
        wall = value.replace(tzinfo=None, fold=0)
        node = {"k": "datetime", "v": wall.isoformat(), "f": value.fold, "z": _timezone_to_json(value.tzinfo)}
        if isinstance(value.tzinfo, pytz.BaseTzInfo) and value.tzinfo.zone is not None:
            variants = _pytz_variants(wall, pytz.timezone(value.tzinfo.zone))
            position = next((i for i, v in enumerate(variants) if v is value.tzinfo), None)
            if position is None:
                raise UnencodableValue(f"{value!r} carries a zone variant pytz would not give that wall time")
            node["p"] = position
        return node
    if isinstance(value, datetime.date):  # after `datetime`, which subclasses it
        return {"k": "date", "v": value.isoformat()}
    if isinstance(value, datetime.time):
        if value.tzinfo is not None:
            raise UnencodableValue(f"{value!r}: a time's zone cannot be stored inside a dataframe cell")
        return {"k": "time", "v": value.isoformat(), "f": value.fold}
    if isinstance(value, pd.Timedelta):
        return {"k": "timedelta", "v": int(value.asm8.astype("int64")), "u": value.unit}
    if isinstance(value, datetime.timedelta):
        return {"k": "pytimedelta", "v": [value.days, value.seconds, value.microseconds]}
    if isinstance(value, bytes):
        return {"k": "bytes", "v": base64.b64encode(value).decode("ascii")}
    if isinstance(value, decimal.Decimal):
        return {"k": "decimal", "v": str(value)}
    if isinstance(value, uuid.UUID):
        return {"k": "uuid", "v": str(value)}
    import shapely

    if isinstance(value, shapely.Geometry):
        return {"k": "wkb", "v": value.wkb_hex}
    raise UnencodableValue(f"{type(value).__name__} cannot be stored inside a dataframe cell")


def _timestamp_from_json(epoch: int, unit: Any, fold: int, zone: dict[str, Any] | None) -> pd.Timestamp:
    import numpy as np

    timestamp = pd.Timestamp(np.datetime64(epoch, unit))
    if zone is None:
        return timestamp.replace(fold=1) if fold else timestamp
    return timestamp.tz_localize(datetime.timezone.utc).tz_convert(_timezone_from_json(zone))


def _datetime_from_json(text: str, fold: int, zone: dict[str, Any] | None, variant: int | None) -> datetime.datetime:
    wall = datetime.datetime.fromisoformat(text)
    if zone is None:
        return wall.replace(fold=fold)
    tz = _timezone_from_json(zone)
    if variant is not None:
        tz = _pytz_variants(wall, tz)[variant]
    return wall.replace(tzinfo=tz, fold=fold)


def _cell_from_json(node: dict[str, Any]) -> Any:
    import numpy as np

    match node:
        case {"k": "j", "v": value}:
            return value
        case {"k": "np", "d": dtype, "v": item}:
            return np.dtype(dtype).type(_cell_from_json(item))
        case {"k": "na"}:
            return pd.NA
        case {"k": "complex", "v": [real, imag]}:
            return complex(real, imag)
        case {"k": "list", "v": items}:
            return [_cell_from_json(n) for n in items]
        case {"k": "tuple", "v": items}:
            return tuple(_cell_from_json(n) for n in items)
        case {"k": "ndarray", "d": dtype, "s": shape, "v": items}:
            return np.array(_cell_from_json(items), dtype=dtype).reshape(shape)
        case {"k": "set", "v": items}:
            return {_cell_from_json(n) for n in items}
        case {"k": "frozenset", "v": items}:
            return frozenset(_cell_from_json(n) for n in items)
        case {"k": "dict", "v": items}:
            return {k: _cell_from_json(n) for k, n in items.items()}
        case {"k": "pairs", "v": items}:
            return {_cell_from_json(k): _cell_from_json(v) for k, v in items}
        case {"k": "date", "v": text}:
            return datetime.date.fromisoformat(text)
        case {"k": "time", "v": text, "f": fold}:
            return datetime.time.fromisoformat(text).replace(fold=fold)
        case {"k": "bytes", "v": text}:
            return base64.b64decode(text)
        case {"k": "decimal", "v": text}:
            return decimal.Decimal(text)
        case {"k": "uuid", "v": text}:
            return uuid.UUID(text)
        case {"k": "wkb", "v": text}:
            import shapely

            return shapely.from_wkb(bytes.fromhex(text))
        case {"k": "nat"}:
            return pd.NaT
        case {"k": "timestamp", "v": epoch, "u": unit, "f": fold, "z": zone}:
            return _timestamp_from_json(epoch, unit, fold, zone)
        case {"k": "datetime", "v": text, "f": fold, "z": zone, **rest}:
            return _datetime_from_json(text, fold, zone, rest.get("p"))
        case {"k": "timedelta", "v": count, "u": unit}:
            return pd.Timedelta(np.timedelta64(count, unit))
        case {"k": "pytimedelta", "v": [days, seconds, microseconds]}:
            return datetime.timedelta(days=days, seconds=seconds, microseconds=microseconds)
        case _:
            raise ValueError(f"unreadable cell node {node!r}")


def encode_cell(value: Any) -> str:
    return json.dumps(_cell_to_json(value))


# The node kinds a Studio data set may hold: tables outlive the version that
# wrote them, and an older reader refuses any other kind.
_DATA_SET_CELL_KINDS = frozenset(
    {"j", "list", "tuple", "dict", "nat", "timestamp", "datetime", "timedelta", "pytimedelta"}
)


def _data_set_kinds_only(node: dict[str, Any]) -> None:
    if node["k"] not in _DATA_SET_CELL_KINDS:
        raise UnencodableValue(f"a {node['k']!r} cell cannot be stored in a data set")
    if node["k"] in ("list", "tuple"):
        for child in node["v"]:
            _data_set_kinds_only(child)
    elif node["k"] == "dict":
        for child in node["v"].values():
            _data_set_kinds_only(child)


def encode_data_set_cell(value: Any) -> str:
    """`encode_cell`, limited to the node kinds every data set reader knows."""
    node = _cell_to_json(value)
    _data_set_kinds_only(node)
    return json.dumps(node)


def decode_cell(text: str) -> Any:
    return _cell_from_json(json.loads(text))


def object_series(values: list[Any], index: Any) -> pd.Series:
    """Build an object-dtype Series without pandas unpacking the sequences."""
    import numpy as np
    import pandas as pd

    array = np.empty(len(values), dtype=object)
    for i, v in enumerate(values):
        array[i] = v
    return pd.Series(array, index=index, dtype=object)


def _parquet_keeps_categories(dtype: pd.CategoricalDtype) -> bool:
    """Arrow reads a dictionary back as a categorical only when its values are strings."""
    return dtype.categories.dtype == object and dtype.categories.inferred_type == "string"


def _to_parquet(column: pd.Series) -> tuple[Any, dict[str, Any]]:
    """What to write in place of `column` (None to write it as is), and what `_from_parquet` needs to undo that."""
    spec: dict[str, Any] = {}
    replacement: Any = None
    dtype = column.dtype
    if isinstance(dtype, pd.SparseDtype):
        spec["astype"] = str(dtype)
        column = column.sparse.to_dense()
        replacement, dtype = column.array, column.dtype
    if isinstance(dtype, pd.ArrowDtype) and pa.types.is_nested(dtype.pyarrow_dtype):
        # pandas writes these but cannot read its own metadata for them back.
        raise UnencodableValue(f"nested Arrow dtype {dtype} cannot be read back from parquet")
    if isinstance(dtype, pd.CategoricalDtype) and not _parquet_keeps_categories(dtype):
        categories = encode_dataframe(pd.DataFrame({"c": dtype.categories}))
        spec["categories"] = base64.b64encode(categories).decode("ascii")
        spec["ordered"] = bool(dtype.ordered)
        return column.cat.codes.to_numpy(), spec
    if dtype.kind == "c":
        spec.setdefault("astype", str(dtype))  # a sparse column's dtype, if it was one
        spec["tree"] = True
        return [encode_cell(complex(v)) for v in column], spec
    if dtype.kind in "mM":
        if column.dt.unit == "s":  # parquet's coarsest unit is ms
            spec["unit"] = "s"
        # pandas reads every zone back from parquet as pytz.
        if (tz := getattr(dtype, "tz", None)) is not None and not isinstance(tz, pytz.BaseTzInfo):
            spec["tz"] = _timezone_to_json(tz)
    if isinstance(dtype, pd.StringDtype) and (dtype.storage != "python" or dtype.na_value is not pd.NA):
        spec["string"] = [dtype.storage, dtype.na_value is not pd.NA]
    if dtype == object:
        if _frame_tree_column(column):
            spec["tree"] = True
            return [encode_cell(v) for v in column], spec
        if (kinds := _missing_kinds(column)) and kinds != ["none"]:
            spec["missing"] = kinds[0]
    return replacement, spec


def _frame_tree_column(column: pd.Series) -> bool:
    """An object column parquet would not read back cell for cell.

    Wider than `tree_columns`: a column of numbers comes back numeric, a column
    of sets as arrays, NaN beside `None` as all `None`, and a `time` without its
    zone. Exact types, since Arrow reads a `str` subclass back as `str`.
    """
    types = _cell_types(column)
    return len(types) > 1 or any(t not in (str, bytes, datetime.date) for t in types) or len(_missing_kinds(column)) > 1


def _from_parquet(column: pd.Series, spec: dict[str, Any]) -> Any:
    import numpy as np

    if spec.get("tree"):
        column = object_series([decode_cell(v) for v in column], column.index)
    if "categories" in spec:
        categories = pd.Index(decode_dataframe(base64.b64decode(spec["categories"]))["c"].array)
        dtype = pd.CategoricalDtype(categories, ordered=spec["ordered"])
        column = pd.Series(pd.Categorical.from_codes(column.to_numpy(), dtype=dtype), index=column.index)
    if (kind := spec.get("missing")) is not None:
        values = column.to_numpy(dtype=object, copy=True)
        values[np.asarray(pd.isna(values))] = {"nan": np.nan, "na": pd.NA}[kind]
        column = object_series(list(values), column.index)
    if (unit := spec.get("unit")) is not None:
        column = column.dt.as_unit(unit)
    if (zone := spec.get("tz")) is not None:
        column = column.dt.tz_convert(_timezone_from_json(zone))
    if (string := spec.get("string")) is not None:
        storage, nan = string
        column = column.astype(pd.StringDtype(storage, na_value=np.nan if nan else pd.NA))
    if (dtype := spec.get("astype")) is not None:
        column = column.astype(pd.api.types.pandas_dtype(dtype))
    return column


def _index_to_parquet(index: pd.Index) -> tuple[pd.Index | None, dict[str, Any]]:
    """The index to write in place of `index` (None to write it as is), and what `_index_from_parquet` needs."""
    if isinstance(index, pd.RangeIndex):  # `index=None` keeps it as start/stop/step
        return None, {}
    spec: dict[str, Any] = {}
    arrays = []
    for i in range(index.nlevels):
        level = index.get_level_values(i)
        values, level_spec = _to_parquet(pd.Series(level))
        if isinstance(level.dtype, pd.StringDtype):  # an index level, unlike a column, reads back as object
            level_spec.setdefault("string", [level.dtype.storage, level.dtype.na_value is not pd.NA])
        arrays.append(level if values is None else values)
        if level_spec:
            spec.setdefault("levels", {})[str(i)] = level_spec
    names = list(index.names)
    if any(n is not None and not isinstance(n, str) for n in names):
        spec["names"] = [_cell_to_json(n) for n in names]
    if isinstance(index, (pd.DatetimeIndex, pd.TimedeltaIndex)) and index.freq is not None:
        spec["freq"] = index.freqstr
    if "levels" not in spec and "names" not in spec:
        return None, spec
    # pandas would write a non-string name as its `str` anyway, with a warning.
    written = [n if n is None or isinstance(n, str) else str(n) for n in names]
    if isinstance(index, pd.MultiIndex):
        return pd.MultiIndex.from_arrays(arrays, names=written), spec
    return pd.Index(arrays[0], name=written[0], tupleize_cols=False), spec


def _index_from_parquet(index: pd.Index, spec: dict[str, Any]) -> pd.Index:
    if levels := spec.get("levels"):
        arrays = [
            _from_parquet(pd.Series(index.get_level_values(i)), levels[str(i)])
            if str(i) in levels
            else index.get_level_values(i)
            for i in range(index.nlevels)
        ]
        if isinstance(index, pd.MultiIndex):
            index = pd.MultiIndex.from_arrays(arrays, names=index.names)
        else:
            index = pd.Index(arrays[0], name=index.name, tupleize_cols=False)
    if (names := spec.get("names")) is not None:
        index = index.set_names([_cell_from_json(n) for n in names])
    if (freq := spec.get("freq")) is not None:
        index = type(index)(index, freq=freq)
    return index


def _labels_are_strings(labels: pd.Index) -> bool:
    if any(n is not None and not isinstance(n, str) for n in labels.names):
        return False
    if isinstance(labels, pd.MultiIndex):
        return all(isinstance(part, str) for label in labels for part in label)
    return all(isinstance(label, str) for label in labels)


def encode_dataframe(df: AnyDataFrame) -> bytes:
    """Encode a frame as (geo)parquet, index included.

    `_persist_df` writes `index=False` because its output is an export; a cache
    entry has to round-trip, and `merge_df` produces a MultiIndex that
    `index=False` silently discards. `index=None` stores a `RangeIndex` as
    start/stop/step, so it reads back as one.

    What parquet cannot carry as is (see `_to_parquet`) is rewritten into
    something it can, and recorded under `FRAME_SPEC_KEY` for decode to undo.

    Raises:
        UnencodableValue: If a structural cell holds something JSON cannot carry,
            which leaves the step uncached rather than stored lossily.
    """
    import geopandas as gpd  # type: ignore[import-untyped]

    is_geo = isinstance(df, gpd.GeoDataFrame)
    frame = df.copy(deep=False)
    spec: dict[str, Any] = {}

    if isinstance(df.columns, pd.RangeIndex):
        spec["column_range"] = [df.columns.start, df.columns.stop, df.columns.step]
    elif df.columns.has_duplicates or not _labels_are_strings(df.columns):
        # Parquet refuses duplicate names and pandas writes others as their `str`,
        # so write positional names and keep the labels.
        spec["labels"] = {
            "values": [_cell_to_json(c) for c in df.columns],
            "names": [_cell_to_json(n) for n in df.columns.names],
            "multi": isinstance(df.columns, pd.MultiIndex),
        }
        frame.columns = [f"__ecoscope_column_{i}__" for i in range(df.shape[1])]
        if is_geo and df._geometry_column_name is not None:
            # geopandas does not follow a relabelling, so point it at the new name.
            active = df.columns.get_loc(df._geometry_column_name)
            spec["labels"]["active"] = active
            frame = frame.set_geometry(frame.columns[active])

    columns = {}
    for i in range(frame.shape[1]):
        values, column_spec = _to_parquet(frame.iloc[:, i])
        if values is not None:
            frame.isetitem(i, values)
        if column_spec:
            columns[str(i)] = column_spec
    if columns:
        spec["columns"] = columns

    index, index_spec = _index_to_parquet(df.index)
    if index is not None:
        frame.index = index
    if index_spec:
        spec["index"] = index_spec

    if not df.flags.allows_duplicate_labels:
        spec["flags"] = False
    if df.attrs:
        # geopandas writes no attrs, and pandas only JSON-native ones.
        spec["attrs"] = _cell_to_json(dict(df.attrs))
        frame.attrs = {}

    buffer = io.BytesIO()
    if has_geometry(frame):
        if not is_geo:
            spec["geo"] = False
            first = next(n for n, dt in zip(frame.columns, frame.dtypes) if isinstance(dt, gpd.array.GeometryDtype))
            frame = gpd.GeoDataFrame(frame, geometry=first)
        frame.to_parquet(buffer, index=None)
    else:
        if is_geo:
            spec["geo"] = True
            frame = pd.DataFrame(frame)
        frame.to_parquet(buffer, index=None)
    data = buffer.getvalue()
    return _with_schema_metadata(data, {FRAME_SPEC_KEY: json.dumps(spec).encode()}) if spec else data


def _with_schema_metadata(parquet: bytes, extra: dict[bytes, bytes]) -> bytes:
    """Rewrite `parquet` with `extra` merged into its schema metadata.

    A read-and-rewrite rather than a write-time option, because neither
    `DataFrame.to_parquet` nor geopandas' geoparquet writer exposes one, and
    reimplementing the geoparquet writer to get at it would be a far larger
    thing to keep correct. Only frames with something to restore pay for it.
    """
    import pyarrow.parquet as pq  # type: ignore[import-untyped]

    table = pq.read_table(io.BytesIO(parquet))
    merged = dict(table.schema.metadata or {}) | extra
    buffer = io.BytesIO()
    pq.write_table(table.replace_schema_metadata(merged), buffer)
    return buffer.getvalue()


def decode_dataframe(data: bytes) -> pd.DataFrame:
    """Read back what `encode_dataframe` wrote."""
    import geopandas as gpd  # type: ignore[import-untyped]
    import pandas as pd
    import pyarrow.parquet as pq  # type: ignore[import-untyped]

    try:
        df: pd.DataFrame = gpd.read_parquet(io.BytesIO(data))
    except ValueError:
        # No geometry column: geopandas refuses, pandas doesn't care.
        df = pd.read_parquet(io.BytesIO(data))

    metadata = pq.read_schema(io.BytesIO(data)).metadata or {}
    spec = json.loads(metadata.get(FRAME_SPEC_KEY, b"{}"))
    for position, column_spec in spec.get("columns", {}).items():
        i = int(position)
        df.isetitem(i, _from_parquet(df.iloc[:, i], column_spec))
    if (labels := spec.get("labels")) is not None:
        values = [_cell_from_json(v) for v in labels["values"]]
        names = [_cell_from_json(n) for n in labels["names"]]
        df.columns = (
            pd.MultiIndex.from_tuples(values, names=names)
            if labels["multi"]
            else pd.Index(values, name=names[0], tupleize_cols=False)
        )
        if (active := labels.get("active")) is not None:
            df = df.set_geometry(df.columns[active])
    if (bounds := spec.get("column_range")) is not None:
        df.columns = pd.RangeIndex(*bounds, name=df.columns.name)
    if (index_spec := spec.get("index")) is not None:
        df.index = _index_from_parquet(df.index, index_spec)
    if (geo := spec.get("geo")) is not None:
        df = gpd.GeoDataFrame(df) if geo else pd.DataFrame(df)
    if (attrs := spec.get("attrs")) is not None:
        df.attrs = _cell_from_json(attrs)
    if spec.get("flags") is False:
        df = df.set_flags(allows_duplicate_labels=False)
    return df


def _encode_frame(df: AnyDataFrame) -> bytes:
    """`encode_dataframe`, which under `WT_CACHE__STRICT` also declines a frame that reads back with another digest.

    The net under `_to_parquet` for shapes it does not know yet. Strict only, as
    in tests and CI: the extra decode and digests about triple a write.
    """
    data = encode_dataframe(df)
    if not strict():
        return data
    try:
        same = digest_dataframe(decode_dataframe(data)) == digest_dataframe(df)
    except Exception as e:
        raise UnencodableValue(f"cannot read this frame back: {e!r}") from e
    if not same:
        raise UnencodableValue("this frame does not read back as it was written")
    return data


def _frame_digest(df: AnyDataFrame) -> str:
    """`digest_dataframe`, but declining instead of taking the run down.

    `persist_df` wants the `TypeError`; the cache wants the step to run and not
    be stored.
    """
    try:
        return digest_dataframe(df)
    except Exception as e:
        raise UnencodableValue(f"cannot content-address this frame: {e}") from e


# Claiming `pd.DataFrame` covers `GeoDataFrame` and pandera's alias, which is
# why this accepts subclasses. The two frame formats share no content-id space,
# because ids are namespaced by codec, so the same data as a frame and as a
# table digests differently. Cross-format dedup is the price of not having to
# agree on a canonical form across two type systems.
FRAME = Codec(
    codec_id=FRAME_CODEC_ID,
    media_type="application/vnd.apache.parquet",
    types=(pd.DataFrame,),
    encode=_encode_frame,
    decode=decode_dataframe,
    digest=_frame_digest,
)


# --------------------------------------------------------------------------- #
# arrow tables
# --------------------------------------------------------------------------- #


def digest_table(table: pa.Table) -> str:
    """Return a full sha256 over an Arrow table's schema and column buffers.

    Not the IPC bytes: buffer padding, alignment and chunking are not part of
    the table's identity but do change the stream, and a digest that moves
    invalidates every downstream step on every run. Chunks are combined first
    so that how a table was assembled doesn't show up in its identity.

    Schema metadata — where the role contract and `ARROW:extension:name` live —
    is part of the identity, since it changes how the data is read.
    """
    combined = table.combine_chunks()
    digest = hashlib.sha256()
    digest.update(combined.schema.to_string(show_field_metadata=True, show_schema_metadata=True).encode())
    for column in combined.columns:
        for chunk in column.chunks:
            for buffer in chunk.buffers():
                if buffer is None:
                    digest.update(b"\x00")
                else:
                    digest.update(struct.pack(">Q", len(buffer)))
                    digest.update(buffer.to_pybytes())
    return digest.hexdigest()


# Arrow promotes a field carrying `ARROW:extension:name` into the named
# extension type when reading IPC -- but only if that extension is registered,
# and geoarrow registers itself on import. So a table holding the name as plain
# metadata reads back as `binary` or as `extension<geoarrow.wkb>` depending on
# what else the process happened to import first, which is `hit != miss` decided
# by import order. Escaping the key on the way out fingerprints the *declared*
# schema, which is the only one that is stable.
_EXTENSION_KEYS = (b"ARROW:extension:name", b"ARROW:extension:metadata")
_ESCAPE_PREFIX = b"ecoscope:escaped:"


def _reschema(table: pa.Table, escape: bool) -> pa.Table:
    import pyarrow as pa

    def rekey(key: bytes) -> bytes:
        if escape:
            return _ESCAPE_PREFIX + key if key in _EXTENSION_KEYS else key
        return key[len(_ESCAPE_PREFIX) :] if key.startswith(_ESCAPE_PREFIX) else key

    fields = []
    changed = False
    for field in table.schema:
        metadata = field.metadata or {}
        # A real extension type carries its name in the type, not the metadata;
        # leave those alone, they round-trip exactly.
        if isinstance(field.type, pa.ExtensionType) or not any(rekey(k) != k for k in metadata):
            fields.append(field)
            continue
        changed = True
        fields.append(field.with_metadata({rekey(k): v for k, v in metadata.items()}))

    if not changed:
        return table
    schema = pa.schema(fields, metadata=table.schema.metadata)
    return pa.Table.from_arrays(list(table.columns), schema=schema)


def encode_table(table: pa.Table) -> bytes:
    import pyarrow as pa

    table = _reschema(table, escape=True)
    buffer = io.BytesIO()
    with pa.ipc.new_stream(buffer, table.schema) as writer:
        writer.write_table(table)
    return buffer.getvalue()


def decode_table(data: bytes) -> pa.Table:
    import pyarrow as pa

    with pa.ipc.open_stream(io.BytesIO(data)) as reader:
        return _reschema(reader.read_all(), escape=False)


ARROW = Codec(
    codec_id=ARROW_CODEC_ID,
    media_type="application/vnd.apache.arrow.stream",
    types=(pa.Table,),
    encode=encode_table,
    decode=decode_table,
    digest=digest_table,
)


# --------------------------------------------------------------------------- #
# keyed iterables
# --------------------------------------------------------------------------- #

# `CompositeFilter` spelled out structurally rather than left to a `dynamic`
# node: JSON has no tuple, so a dynamic key decodes as a list, and both
# `groupbykey` and `align_keyed_iterable_to_reference` use the key as a dict key.
_FILTER = {"k": "tuple", "of": [{"k": "json"}, {"k": "json"}, {"k": "json"}]}
_COMPOSITE_FILTER = {"k": "list", "tuple": True, "of": _FILTER}
# `groupbykey_passthrough_skip` emits a null key for a slot it had nothing to key on.
_KEY = {"k": "union", "of": [_COMPOSITE_FILTER, {"k": "json"}]}

_KEYED_LEAF = [
    {"k": "ref", "codec": FRAME},
    {"k": "ref", "codec": ARROW},
    {"k": "ref", "codec": SKIP},
]
# `groupbykey` collects its inputs into a list per key, so one level of nesting
# is part of the contract (`CombinedKeyedIterable`). The list arm precedes
# `dynamic` so that a list holding a frame is walked rather than refused.
_KEYED_VALUE = {
    "k": "union",
    "of": [
        *_KEYED_LEAF,
        {"k": "list", "of": {"k": "union", "of": [*_KEYED_LEAF, {"k": "dynamic"}]}},
        {"k": "dynamic"},
    ],
}

KEYED_ANY = Tree(
    {"k": "list", "of": {"k": "tuple", "of": [_KEY, _KEYED_VALUE]}},
    types=(list,),
    codec_id=KEYED_CODEC_ID,
    media_type="application/vnd.ecoscope.keyed-iterable",
)
"""The codec for the two keyed aliases whose value is `Any`.

The only place a leaf table survives: `KeyedIterableOfAny` and
`CombinedKeyedIterable` say nothing about what they carry, so the arms here
are the declaration. Everything else resolves from its own return type.
"""


# --------------------------------------------------------------------------- #
# pydantic models
# --------------------------------------------------------------------------- #


def _model_json(value: BaseModel) -> bytes:
    """Canonical JSON for a model: one form per value, so the digest can hash it.

    `warnings="error"`: pydantic's default for a value it cannot represent is
    to warn and emit it anyway, which is the silent stringification this whole
    module exists to avoid.
    """
    try:
        dumped = value.model_dump(mode="json", warnings="error")
    except Exception as e:
        raise UnencodableValue(f"{type(value).__name__} cannot be dumped losslessly: {e}") from e
    return json.dumps(dumped, sort_keys=True).encode("utf-8")


def model_codec(cls: type[BaseModel]) -> Codec:
    """The codec every `CacheableModel` declares.

    Refuses subclasses, and is keyed by class in the codec id, because the
    payload no longer names the class to rebuild: a subclass encoded through
    its parent's codec would come back narrowed, and a declaration that moved
    has to miss rather than decode as something else.
    """
    return Codec(
        codec_id=f"{MODEL_CODEC_ID}/{cls.__module__}.{cls.__qualname__}",
        media_type="application/json",
        types=(cls,),
        encode=_model_json,
        decode=cls.model_validate_json,
        digest=lambda v: hashlib.sha256(_model_json(v)).hexdigest(),
        accepts_subclasses=False,
    )


# --------------------------------------------------------------------------- #
# groupers
# --------------------------------------------------------------------------- #

# The model fields each grouper's canonical state is known to cover. Compared
# as a set on encode, so a field added to one of these models refuses the value
# rather than silently dropping it from both the payload and the digest --
# which would serve one variant's cache entry for another's.
_GROUPER_FIELDS: dict[str, tuple[str, ...]] = {
    "ValueGrouper": ("index_name", "help_text"),
    "TemporalGrouper": ("temporal_index", "help_text", "is_temporal"),
    "SpatialGrouper": ("spatial_index_name", "help_text"),
}

# `(class name, fields, resolved regions)`. A fixed tuple rather than a dict so
# that the frame's position is part of the structure rather than of the value.
_GROUPER_STATE = Tree(
    {
        "k": "tuple",
        "of": [
            {"k": "json"},
            {"k": "dict", "key": {"k": "json"}, "of": {"k": "json"}},
            {"k": "union", "of": [{"k": "ref", "codec": FRAME}, {"k": "json"}]},
        ],
    }
)


def _grouper_state(grouper: Any) -> tuple[str, dict[str, Any], Any]:
    """A grouper's identity as a tuple `_GROUPER_STATE` can carry.

    Built by hand rather than from `model_dump`, which is wrong here twice
    over: `help_text` and `is_temporal` are declared `exclude=True` and so
    never appear, and `temporal_index` dumps as a dataclass holding a bound
    `list.index` method, which is neither JSON nor stable across processes.
    The directive string it validates from is the whole of its identity.
    """
    from ecoscope.platform.indexes import SpatialGrouper, TemporalGrouper, ValueGrouper

    name = type(grouper).__name__
    if set(type(grouper).model_fields) != set(_GROUPER_FIELDS.get(name, ())):
        raise UnencodableValue(f"{name} has fields this codec does not cover")

    if isinstance(grouper, TemporalGrouper):
        fields: dict[str, Any] = {
            "temporal_index": grouper.temporal_index.directive,
            "help_text": grouper.help_text,
        }
    elif isinstance(grouper, SpatialGrouper):
        fields = {"spatial_index_name": grouper.spatial_index_name, "help_text": grouper.help_text}
    elif isinstance(grouper, ValueGrouper):
        fields = {"index_name": grouper.index_name, "help_text": grouper.help_text}
    else:
        raise UnencodableValue(f"{name} is not a user-defined grouper")

    # The resolved regions live in a `PrivateAttr`, and `SpatialGrouper` hashes
    # and compares on `spatial_index_name` alone -- so a resolved and an
    # unresolved instance are indistinguishable to everything except this.
    # Carried explicitly, as a frame leaf, or a hit would hand
    # `add_spatial_index` a grouper with nothing to join against.
    resolved = grouper.spatial_regions if isinstance(grouper, SpatialGrouper) else None
    return name, fields, resolved


def _grouper_from_state(cls: type, state: tuple[str, dict[str, Any], Any]) -> Any:
    name, fields, resolved = state
    if name != cls.__name__:
        raise ValueError(f"payload holds a {name}, not a {cls.__name__}")
    grouper = cls(**fields)
    if resolved is not None:
        grouper.resolve(resolved)
    return grouper


def grouper_codec(cls: type) -> Codec:
    """The codec for one of the three user-defined grouper classes.

    `AllGrouper` is deliberately not one of them: it excludes no field and
    holds no private attribute, so it round-trips through `model_codec` like
    any other config model.
    """
    return Codec(
        codec_id=f"{GROUPER_CODEC_ID}/{cls.__module__}.{cls.__qualname__}",
        media_type="application/vnd.ecoscope.grouper",
        types=(cls,),
        encode=lambda v: _GROUPER_STATE.encode(_grouper_state(v)),
        decode=lambda d: _grouper_from_state(cls, _GROUPER_STATE.decode(d)),
        digest=lambda v: _GROUPER_STATE.digest(_grouper_state(v)),
        accepts_subclasses=False,
    )
