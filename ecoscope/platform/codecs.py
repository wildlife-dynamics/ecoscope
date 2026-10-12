"""How an `ecoscope.platform` task return becomes bytes, and back again.

Registered with `wt-task` through the `wt_task.codecs` entry point, so the step
cache can address dataframes, Arrow tables, groupers and keyed iterables — a
value no codec claims is not merely unstored, it is undigestible, which makes
every step below it uncacheable too.

The digest half is also what `persist_df` and friends use for their
content-addressed filenames, so there is one content-identity implementation
rather than one per caller.

Deliberately a module, not a package: `[tool.setuptools].packages` in
`pyproject.toml` is an explicit list, so a new subpackage would silently not
ship.
"""

from __future__ import annotations

import functools
import hashlib
import importlib
import io
import json
import struct
import sys
from typing import TYPE_CHECKING, Any, Callable

from wt_task.cache import Codec, get_registry

if TYPE_CHECKING:
    import pandas as pd
    import pyarrow as pa

    from ecoscope.platform.annotations import AnyDataFrame

FRAME_CODEC_ID = "ecoscope.frame/v1"
ARROW_CODEC_ID = "ecoscope.arrow/v1"
KEYED_CODEC_ID = "ecoscope.keyed/v1"
GROUPER_CODEC_ID = "ecoscope.grouper/v1"
GROUPERS_CODEC_ID = "ecoscope.groupers/v1"
MODEL_CODEC_ID = "ecoscope.model/v1"
MODELS_CODEC_ID = "ecoscope.models/v1"
GUARD_CODEC_ID = "ecoscope.guard/v1"

# Returned by a `digest` that could not identify its value. The contract gives
# `digest` no way to decline, and raising from it would abort the whole run
# (`_content_id` is called outside wt-task's try/except), so the refusal is
# routed through `encode` instead: it raises, wt-task warns, and the step is
# simply not cached. Never used as a real content id.
UNDIGESTIBLE = "undigestible"


# --------------------------------------------------------------------------- #
# dataframes
# --------------------------------------------------------------------------- #

# Schema-metadata key naming the object columns `encode_dataframe` rewrote as
# JSON trees. Arrow types such a column structurally, which loses information in
# two directions: a `list` and a `tuple` cell both come back as `ndarray`, and a
# column of dicts becomes a struct whose schema is the *union* of the keys seen,
# so `{"k": 1}` next to `{"k": 2, "j": "x"}` reads back as `{"k": 1, "j": None}`
# -- which is `hit != miss` on the ER event-details column specifically.
TREE_COLUMNS_KEY = b"ecoscope:tree_columns"


def has_geometry(df: AnyDataFrame) -> bool:
    """Whether `df` carries at least one geometry-dtype column."""
    import geopandas as gpd  # type: ignore[import-untyped]

    return any(isinstance(dtype, gpd.array.GeometryDtype) for dtype in df.dtypes)


def _fallback_hash(obj: Any) -> Any:
    import pandas as pd

    try:
        return pd.util.hash_pandas_object(obj, index=False).values
    except (TypeError, ValueError, AttributeError):
        # Cells pandas can't hash (lists/dicts in event details). `repr`
        # rather than `str`: in a mixed column `str` collapses 1 and "1"
        # to the same text, and a filename collision serves wrong bytes.
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
    for name, dtype in zip(df.columns, df.dtypes):
        if isinstance(dtype, gpd.array.GeometryDtype):
            column_crs = df[name].crs
            crs[str(name)] = None if column_crs is None else column_crs.to_string()

    index = df.index
    return {
        "index_names": [None if n is None else str(n) for n in index.names],
        "index_dtypes": [str(index.get_level_values(i).dtype) for i in range(index.nlevels)],
        "crs": crs,
        "attrs": {str(k): v for k, v in df.attrs.items()},
    }


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
    try:
        digest.update(np.ascontiguousarray(pd.util.hash_pandas_object(df).values))
    except (TypeError, ValueError, AttributeError):
        # Only now, and only for the columns that genuinely fail, pay per-column
        # costs. Every row still contributes; hashing a `head(5)` sample here
        # let two frames sharing shape and first rows collide.
        for obj in (df.index, *(df.iloc[:, i] for i in range(len(df.columns)))):
            digest.update(np.ascontiguousarray(_fallback_hash(obj)))
    digest.update(json.dumps([[str(c), str(dt)] for c, dt in zip(df.columns, df.dtypes)]).encode())
    digest.update(json.dumps(_frame_metadata(df), sort_keys=True, default=str).encode())
    return digest.hexdigest()


def _tree_columns(df: AnyDataFrame) -> list[str]:
    """Object columns holding a `list`, `tuple` or `dict` in any cell.

    One rule for all three rather than per-kind repair on the way out: the
    structure is what Arrow flattens, and reconstructing it from a typed column
    means guessing, which is how a column of genuine lists becomes a column of
    tuples.
    """
    found = []
    for position, (name, dtype) in enumerate(zip(df.columns, df.dtypes)):
        if dtype == object and any(isinstance(v, (list, tuple, dict)) for v in df.iloc[:, position]):
            found.append(str(name))
    return found


def _refuse_leaf(codec: Codec, value: Any) -> dict[str, Any]:
    raise UnencodableValue(f"{type(value).__name__} cannot be stored inside a dataframe cell")


def _encode_cell(value: Any) -> str:
    return json.dumps(_walk(value, _refuse_leaf))


def _decode_cell(text: str) -> Any:
    return _unwalk(json.loads(text), b"")


def _object_series(values: list[Any], index: Any) -> pd.Series:
    """Build an object-dtype Series without pandas unpacking the sequences."""
    import numpy as np
    import pandas as pd

    array = np.empty(len(values), dtype=object)
    for i, v in enumerate(values):
        array[i] = v
    return pd.Series(array, index=index, dtype=object)


def encode_dataframe(df: AnyDataFrame) -> bytes:
    """Encode a frame as (geo)parquet, index included.

    `_persist_df` writes `index=False` because its output is an export; a cache
    entry has to round-trip, and `merge_df` produces a MultiIndex that
    `index=False` silently discards.

    Raises:
        UnencodableValue: If a structural cell holds something JSON cannot carry,
            which leaves the step uncached rather than stored lossily.
    """
    import geopandas as gpd  # type: ignore[import-untyped]

    trees = _tree_columns(df)
    if trees:
        df = df.copy()
        for name in trees:
            column = df[name]
            df[name] = [_encode_cell(v) for v in column]

    buffer = io.BytesIO()
    if has_geometry(df):
        gpd.GeoDataFrame(df).to_parquet(buffer, index=True)
    else:
        df.to_parquet(buffer, index=True)
    data = buffer.getvalue()

    if not trees:
        return data
    return _with_schema_metadata(data, {TREE_COLUMNS_KEY: json.dumps(trees).encode()})


def _with_schema_metadata(parquet: bytes, extra: dict[bytes, bytes]) -> bytes:
    """Rewrite `parquet` with `extra` merged into its schema metadata.

    A read-and-rewrite rather than a write-time option, because neither
    `DataFrame.to_parquet` nor geopandas' geoparquet writer exposes one, and
    reimplementing the geoparquet writer to get at it would be a far larger
    thing to keep correct. Only frames with sequence cells pay for it.
    """
    import pyarrow.parquet as pq

    table = pq.read_table(io.BytesIO(parquet))
    merged = dict(table.schema.metadata or {}) | extra
    buffer = io.BytesIO()
    pq.write_table(table.replace_schema_metadata(merged), buffer)
    return buffer.getvalue()


def decode_dataframe(data: bytes) -> pd.DataFrame:
    """Read back what `encode_dataframe` wrote."""
    import geopandas as gpd  # type: ignore[import-untyped]
    import pandas as pd
    import pyarrow.parquet as pq

    try:
        df: pd.DataFrame = gpd.read_parquet(io.BytesIO(data))
    except ValueError:
        # No geometry column: geopandas refuses, pandas doesn't care.
        df = pd.read_parquet(io.BytesIO(data))

    metadata = pq.read_schema(io.BytesIO(data)).metadata or {}
    if (raw := metadata.get(TREE_COLUMNS_KEY)) is not None:
        for name in json.loads(raw):
            df[name] = _object_series([_decode_cell(v) for v in df[name]], df.index)
    return df


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


# --------------------------------------------------------------------------- #
# keyed iterables
# --------------------------------------------------------------------------- #

_KEYED_MAGIC = b"ecoscope.keyed/v1\n"


class UnencodableValue(TypeError):
    """Raised while walking a keyed iterable that holds something with no codec."""


def is_keyed_iterable(value: Any) -> bool:
    """Whether `value` has the `list[tuple[CompositeFilter | None, X]]` shape.

    A keyed iterable is a plain `list`, so this codec has to confirm on content.
    Returning False hands the value on to the builtin JSON codec, which is the
    right answer for an ordinary list — and the reason the check is on the key
    shape alone rather than on the values: a keyed iterable whose values turn
    out to be unencodable must still be *claimed* here and refused at encode
    time, because falling through would let `json/v1` stringify a dataframe
    with `default=str` and store the repr as if it were the value.

    Examples:
        >>> is_keyed_iterable([((("month", "=", "January"),), "x")])
        True
        >>> is_keyed_iterable([1, 2, 3])
        False
        >>> is_keyed_iterable([])
        False
    """
    if not isinstance(value, list) or not value:
        return False
    return all(_is_keyed_entry(entry) for entry in value)


def _is_keyed_entry(entry: Any) -> bool:
    if not isinstance(entry, tuple) or len(entry) != 2:
        return False
    key = entry[0]
    if key is None:
        return True
    if not isinstance(key, tuple) or not key:
        return False
    return all(isinstance(f, tuple) and len(f) == 3 and isinstance(f[0], str) and f[1] == "=" for f in key)


def _walk(value: Any, leaf: Callable[[Codec, Any], dict[str, Any]]) -> dict[str, Any]:
    """Describe `value` as a tree, delegating anything that isn't structure.

    Containers are walked rather than handed to the registry because `json/v1`
    claims `list` and `dict` unconditionally and would stringify whatever is
    inside them. `leaf` is what separates digesting from encoding: the two
    share this walk so they can never disagree about what is encodable.
    """
    if value is None or isinstance(value, (str, bool, int, float)):
        return {"k": "j", "v": value}
    if isinstance(value, (list, tuple)):
        return {"k": "list" if isinstance(value, list) else "tuple", "v": [_walk(v, leaf) for v in value]}
    if isinstance(value, dict):
        if not all(isinstance(k, str) for k in value):
            raise UnencodableValue(f"dict with non-string keys: {list(value)[:3]!r}")
        return {"k": "dict", "v": {k: _walk(v, leaf) for k, v in value.items()}}
    codec = get_registry().for_value(value)
    if codec is None:
        raise UnencodableValue(f"no codec claims {type(value).__name__}")
    return leaf(codec, value)


def _unwalk(node: dict[str, Any], blobs: bytes) -> Any:
    kind = node["k"]
    if kind == "j":
        return node["v"]
    if kind == "list":
        return [_unwalk(n, blobs) for n in node["v"]]
    if kind == "tuple":
        return tuple(_unwalk(n, blobs) for n in node["v"])
    if kind == "dict":
        return {k: _unwalk(n, blobs) for k, n in node["v"].items()}
    codec = get_registry().for_id(node["id"])
    if codec is None:
        raise ValueError(f"codec {node['id']!r} is no longer installed")
    return codec.decode(blobs[node["o"] : node["o"] + node["n"]])


def encode_tree(value: Any, magic: bytes) -> bytes:
    """Encode as a JSON header of structure plus a concatenation of leaf blobs.

    The header carries the structure because a `CompositeFilter` is a tuple and
    so neither a JSON object key nor distinguishable from a list once encoded;
    the blobs stay raw so a frame costs its parquet size, not a third more
    as base64. `magic` names the codec whose shape the header describes, so a
    payload can never be decoded by the wrong one.
    """
    blobs: list[bytes] = []
    offset = 0

    def leaf(codec: Codec, v: Any) -> dict[str, Any]:
        nonlocal offset
        payload = codec.encode(v)
        blobs.append(payload)
        node = {"k": "c", "id": codec.codec_id, "o": offset, "n": len(payload)}
        offset += len(payload)
        return node

    header = json.dumps(_walk(value, leaf)).encode("utf-8")
    return magic + struct.pack(">Q", len(header)) + header + b"".join(blobs)


def decode_tree(data: bytes, magic: bytes) -> Any:
    if not data.startswith(magic):
        raise ValueError(f"payload does not start with {magic!r}")
    start = len(magic)
    (length,) = struct.unpack(">Q", data[start : start + 8])
    body = start + 8
    header = json.loads(data[body : body + length])
    return _unwalk(header, data[body + length :])


def digest_tree(value: Any) -> str:
    """Digest structure plus each leaf's own digest — never the encoded bytes.

    A leaf whose own codec could not identify it takes the whole container down
    with it: `UNDIGESTIBLE` is a single shared marker, so letting it stand in
    for a leaf would make two containers differing only in that leaf digest the
    same.

    Note a `CompositeFilter` key is digested losslessly rather than through
    `_hash_grouper_key`: that encoding drops the operator and truncates to 24
    bits, which is a fine filename and far too weak to carry cache identity.
    """

    def leaf(codec: Codec, v: Any) -> dict[str, Any]:
        if (leaf_digest := codec.digest(v)) == UNDIGESTIBLE:
            raise UnencodableValue(f"{codec.codec_id} cannot content-address this {type(v).__name__}")
        return {"k": "c", "id": codec.codec_id, "d": leaf_digest}

    return hashlib.sha256(json.dumps(_walk(value, leaf), sort_keys=True).encode("utf-8")).hexdigest()


def encode_keyed_iterable(value: Any) -> bytes:
    return encode_tree(value, _KEYED_MAGIC)


def decode_keyed_iterable(data: bytes) -> Any:
    return decode_tree(data, _KEYED_MAGIC)


# --------------------------------------------------------------------------- #
# groupers
# --------------------------------------------------------------------------- #

_GROUPER_MAGIC = b"ecoscope.grouper/v1\n"
_GROUPERS_MAGIC = b"ecoscope.groupers/v1\n"

# The model fields each grouper's canonical state is known to cover. Compared
# as a set on every dispatch, so a field added to one of these models makes the
# codec decline the value rather than silently drop it from both the payload
# and the digest -- which would serve one variant's cache entry for another's.
_GROUPER_FIELDS: dict[str, tuple[str, ...]] = {
    "AllGrouper": ("index_name",),
    "ValueGrouper": ("index_name", "help_text"),
    "TemporalGrouper": ("temporal_index", "help_text", "is_temporal"),
    "SpatialGrouper": ("spatial_index_name", "help_text"),
}


@functools.lru_cache(maxsize=1)
def _grouper_types() -> dict[str, type]:
    from ecoscope.platform import indexes

    return {name: getattr(indexes, name) for name in _GROUPER_FIELDS}


def is_grouper(value: Any) -> bool:
    """Whether `value` is a grouper this codec can round-trip exactly."""
    name = type(value).__name__
    if _grouper_types().get(name) is not type(value):
        return False
    return set(type(value).model_fields) == set(_GROUPER_FIELDS[name])


def is_grouper_list(value: Any) -> bool:
    """Whether `value` is the `UserDefinedGroupers` list `set_groupers` passes on.

    The empty list is deliberately not claimed: `json/v1` already round-trips
    it exactly, and `set_groupers` turns a blank `groupers` parameter into a
    bare `AllGrouper` rather than `[]` anyway.

    Examples:
        >>> from ecoscope.platform.indexes import ValueGrouper
        >>> is_grouper_list([ValueGrouper(index_name="species")])
        True
        >>> is_grouper_list([])
        False
    """
    return isinstance(value, list) and bool(value) and all(is_grouper(v) for v in value)


def _grouper_state(grouper: Any) -> dict[str, Any]:
    """A grouper's identity as a tree the container encoder can carry.

    Built by hand rather than from `model_dump`, which is wrong here twice
    over: `help_text` and `is_temporal` are declared `exclude=True` and so
    never appear, and `temporal_index` dumps as a dataclass holding a bound
    `list.index` method, which is neither JSON nor stable across processes.
    The directive string it validates from is the whole of its identity.
    """
    from ecoscope.platform.indexes import AllGrouper, SpatialGrouper, TemporalGrouper, ValueGrouper

    if isinstance(grouper, TemporalGrouper):
        fields: dict[str, Any] = {
            "temporal_index": grouper.temporal_index.directive,
            "help_text": grouper.help_text,
        }
    elif isinstance(grouper, SpatialGrouper):
        fields = {"spatial_index_name": grouper.spatial_index_name, "help_text": grouper.help_text}
    elif isinstance(grouper, ValueGrouper):
        fields = {"index_name": grouper.index_name, "help_text": grouper.help_text}
    elif isinstance(grouper, AllGrouper):
        fields = {"index_name": grouper.index_name}
    else:
        raise UnencodableValue(f"{type(grouper).__name__} is not a grouper")

    state = {"type": type(grouper).__name__, "fields": fields}
    if isinstance(grouper, SpatialGrouper):
        # The resolved regions live in a `PrivateAttr`, and `SpatialGrouper`
        # hashes and compares on `spatial_index_name` alone -- so a resolved
        # and an unresolved instance are indistinguishable to everything except
        # this. Carried explicitly, as a frame leaf, or a hit would hand
        # `add_spatial_index` a grouper with nothing to join against.
        state["resolved"] = grouper.spatial_regions
    return state


def _grouper_from_state(state: dict[str, Any]) -> Any:
    types = _grouper_types()
    if (cls := types.get(state["type"])) is None:
        raise ValueError(f"unknown grouper type {state['type']!r}")
    grouper = cls(**state["fields"])
    if (resolved := state.get("resolved")) is not None:
        grouper.resolve(resolved)
    return grouper


def encode_grouper(value: Any) -> bytes:
    return encode_tree(_grouper_state(value), _GROUPER_MAGIC)


def decode_grouper(data: bytes) -> Any:
    return _grouper_from_state(decode_tree(data, _GROUPER_MAGIC))


def digest_grouper(value: Any) -> str:
    return digest_tree(_grouper_state(value))


def encode_groupers(value: Any) -> bytes:
    return encode_tree(value, _GROUPERS_MAGIC)


def decode_groupers(data: bytes) -> Any:
    return decode_tree(data, _GROUPERS_MAGIC)


# --------------------------------------------------------------------------- #
# pydantic models
# --------------------------------------------------------------------------- #

_MODELS_MAGIC = b"ecoscope.models/v1\n"


def _model_path(cls: type) -> str:
    return f"{cls.__module__}:{cls.__qualname__}"


def _resolve_model(path: str) -> Any:
    """Find the class a payload names, without widening what the store can run.

    Importing a module is code execution, so a name read back from the cache
    may only reach a module whose top-level package this process has already
    imported -- which every task's own module has, since the DAG imports its
    functions before any step runs.
    """
    module_name, _, qualname = path.partition(":")
    module = sys.modules.get(module_name)
    if module is None:
        if module_name.partition(".")[0] not in sys.modules:
            raise ValueError(f"{path!r} names a package this process has not imported")
        module = importlib.import_module(module_name)

    cls = module
    for part in qualname.split("."):
        cls = getattr(cls, part)
    if not (isinstance(cls, type) and hasattr(cls, "model_validate")):
        raise ValueError(f"{path!r} does not name a pydantic model")
    return cls


def is_roundtrippable_model(value: Any) -> bool:
    """Whether a model can be rebuilt from its own JSON dump, exactly.

    Two independent failure modes, and neither check sees the other's:

    A field the serializer drops -- `exclude=True`, a `PrivateAttr`, a custom
    `@model_serializer` -- is missing from both sides of any value comparison,
    so it can only be caught structurally. That is `Dashboard`, whose
    serializer emits a different shape entirely, and `SpatialGrouper`, whose
    resolved regions hide in a private attribute.

    A field that survives serialization but comes back as something else -- a
    tuple under an `Any` annotation, or a value the model's own validators
    reject on the way back in -- is invisible to any structural check and only
    shows up by doing it. So this pays a dump-validate-dump per dispatch;
    these are config-sized models.
    """
    cls = type(value)
    try:
        decorators = cls.__pydantic_decorators__
        if (
            cls.__private_attributes__
            or decorators.model_serializers
            or decorators.field_serializers
            or any(field.exclude for field in cls.model_fields.values())
        ):
            return False
        _model_path(cls)
        # `warnings="error"`: pydantic's default for a value it cannot
        # represent is to warn and emit it anyway, which is the silent
        # stringification this whole codec exists to avoid.
        dumped = value.model_dump(mode="json", warnings="error")
        return cls.model_validate(dumped).model_dump() == value.model_dump()
    except Exception:
        # `matches` is called bare by the registry, so anything at all here has
        # to read as "not mine" rather than take the run down.
        return False


def is_model_list(value: Any) -> bool:
    """Whether `value` is a list of models this codec can round-trip."""
    if not isinstance(value, list) or not value:
        return False
    from pydantic import BaseModel

    return all(isinstance(v, BaseModel) and is_roundtrippable_model(v) for v in value)


def encode_model(value: Any) -> bytes:
    """Encode as canonical JSON: the class to rebuild, and its dump."""
    payload = {"type": _model_path(type(value)), "data": value.model_dump(mode="json", warnings="error")}
    return json.dumps(payload, sort_keys=True).encode("utf-8")


def decode_model(data: bytes) -> Any:
    payload = json.loads(data)
    return _resolve_model(payload["type"]).model_validate(payload["data"])


def digest_model(value: Any) -> str:
    """Hash the encoded bytes — sound here, and only here, because they are canonical.

    `sort_keys` plus a JSON dump of a declared schema is one form per value,
    unlike parquet, so there is nothing for a second canonicalisation to fix.
    """
    return hashlib.sha256(encode_model(value)).hexdigest()


def encode_models(value: Any) -> bytes:
    return encode_tree(value, _MODELS_MAGIC)


def decode_models(data: bytes) -> Any:
    return decode_tree(data, _MODELS_MAGIC)


# --------------------------------------------------------------------------- #
# the JSON guard
# --------------------------------------------------------------------------- #


def is_json_native(value: Any) -> bool:
    """Whether `json/v1` would return `value` unchanged.

    Tuples count as foreign because JSON has no tuple, and a `dict` with
    non-string keys because JSON has no non-string key.

    Examples:
        >>> is_json_native(["a", 1, {"b": None}])
        True
        >>> is_json_native([("a", 1)])
        False
    """
    if value is None or isinstance(value, (str, bool, int, float)):
        return True
    if isinstance(value, list):
        return all(is_json_native(v) for v in value)
    if isinstance(value, dict):
        return all(isinstance(k, str) and is_json_native(v) for k, v in value.items())
    return False


def _guard_refuse(value: Any) -> Any:
    raise TypeError(
        f"{type(value).__name__} holds values JSON cannot represent; not caching it " f"(see {GUARD_CODEC_ID})"
    )


# `json/v1` claims `list` and `dict` unconditionally and encodes with
# `default=str`, so a value it cannot represent is stringified rather than
# refused -- `gather_dashboard`'s grouper-keyed choices dict, say, would come
# back as a dict of reprs. This sits between the two and turns that into "not
# cached", which is what refusing was supposed to mean.
GUARD_CODEC = Codec(
    codec_id=GUARD_CODEC_ID,
    media_type="application/octet-stream",
    types=(list, dict),
    encode=_guard_refuse,
    decode=_guard_refuse,
    digest=lambda _v: UNDIGESTIBLE,
    matches=lambda v: not is_json_native(v),
    priority=-50,
)


# --------------------------------------------------------------------------- #
# registration
# --------------------------------------------------------------------------- #


def _refusing_digest(digest: Callable[[Any], str]) -> Callable[[Any], str]:
    """Wrap `digest` so a value it cannot identify refuses instead of raising.

    `digest` is called outside wt-task's try/except, so an exception here aborts
    the run rather than skipping the cache -- and `digest_dataframe` does raise,
    deliberately, on cells that are neither hashable nor reprable.
    """

    def wrapped(value: Any) -> str:
        try:
            return digest(value)
        except Exception:
            return UNDIGESTIBLE

    return wrapped


def _refusing_encode(digest: Callable[[Any], str], encode: Callable[[Any], bytes]) -> Callable[[Any], bytes]:
    """Wrap `encode` to refuse exactly what `_refusing_digest` refused.

    Storing a payload under the shared `UNDIGESTIBLE` content id would let two
    unrelated steps read each other's data, so the refusal has to happen here,
    where wt-task turns it into "this step is not cached".
    """

    def wrapped(value: Any) -> bytes:
        if digest(value) == UNDIGESTIBLE:
            raise TypeError(f"cannot content-address {type(value).__name__}; not caching it")
        return encode(value)

    return wrapped


def _codec(codec_id: str, media_type: str, types: tuple[type, ...], digest, encode, decode, **kwargs) -> Codec:
    safe_digest = _refusing_digest(digest)
    return Codec(
        codec_id=codec_id,
        media_type=media_type,
        types=types,
        encode=_refusing_encode(safe_digest, encode),
        decode=decode,
        digest=safe_digest,
        **kwargs,
    )


def codecs() -> list[Codec]:
    """The codecs this distribution contributes to `wt_task.codecs`.

    Imports are deferred to here so that declaring the entry point costs nothing
    until the cache actually looks for it.
    """
    import pandas as pd
    import pyarrow as pa
    from pydantic import BaseModel

    from ecoscope.platform.indexes import AllGrouper, SpatialGrouper, TemporalGrouper, ValueGrouper

    return [
        # Claiming `pd.DataFrame` covers `GeoDataFrame` and pandera's alias by
        # MRO; the two formats share no `content_id` space, because ids are
        # namespaced by codec, so the same data as a frame and as a table
        # digests differently. Cross-format dedup is the price of not having to
        # agree on a canonical form across two type systems.
        _codec(
            FRAME_CODEC_ID,
            "application/vnd.apache.parquet",
            (pd.DataFrame,),
            digest_dataframe,
            encode_dataframe,
            decode_dataframe,
        ),
        _codec(
            ARROW_CODEC_ID,
            "application/vnd.apache.arrow.stream",
            (pa.Table,),
            digest_table,
            encode_table,
            decode_table,
        ),
        _codec(
            KEYED_CODEC_ID,
            "application/vnd.ecoscope.keyed-iterable",
            (list,),
            digest_tree,
            encode_keyed_iterable,
            decode_keyed_iterable,
            matches=is_keyed_iterable,
        ),
        # `set_groupers` returns a bare `AllGrouper` for the ungrouped case and
        # a list otherwise, so both shapes have to be claimed or the grouped
        # subtree stays uncacheable either way. The three `list` claims here --
        # groupers, keyed iterables, the guard -- are disjoint on `matches`;
        # the priorities exist only because equal ones are a registration
        # error, not because the order is ever consulted.
        _codec(
            GROUPER_CODEC_ID,
            "application/vnd.ecoscope.grouper",
            (AllGrouper, ValueGrouper, TemporalGrouper, SpatialGrouper),
            digest_grouper,
            encode_grouper,
            decode_grouper,
            matches=is_grouper,
        ),
        _codec(
            GROUPERS_CODEC_ID,
            "application/vnd.ecoscope.groupers",
            (list,),
            digest_tree,
            encode_groupers,
            decode_groupers,
            matches=is_grouper_list,
            priority=10,
        ),
        # Claiming `BaseModel` reaches the ~57 tasks returning one, but only
        # the models `is_roundtrippable_model` can prove: `TimeRange` and
        # `TimezoneInfo` gate the whole events frame chain, while `Dashboard`
        # and the groupers are declined here and handled, or not, elsewhere.
        # The concrete-class claims above win over this one by MRO, so a
        # grouper never reaches it even though `AllGrouper` would pass.
        _codec(
            MODEL_CODEC_ID,
            "application/json",
            (BaseModel,),
            digest_model,
            encode_model,
            decode_model,
            matches=is_roundtrippable_model,
        ),
        _codec(
            MODELS_CODEC_ID,
            "application/vnd.ecoscope.models",
            (list,),
            digest_tree,
            encode_models,
            decode_models,
            matches=is_model_list,
            priority=5,
        ),
        GUARD_CODEC,
    ]
