"""How an `ecoscope.platform` task return becomes bytes, and back again.

Registered with `wt-task` through the `wt_task.codecs` entry point, so the step
cache can address dataframes, Arrow tables and keyed iterables — a value no
codec claims is not merely unstored, it is undigestible, which makes every step
below it uncacheable too.

The digest half is also what `persist_df` and friends use for their
content-addressed filenames, so there is one content-identity implementation
rather than one per caller.

Deliberately a module, not a package: `[tool.setuptools].packages` in
`pyproject.toml` is an explicit list, so a new subpackage would silently not
ship.
"""

from __future__ import annotations

import hashlib
import io
import json
import struct
from typing import TYPE_CHECKING, Any, Callable

from wt_task.cache import Codec, get_registry

if TYPE_CHECKING:
    import pandas as pd
    import pyarrow as pa

    from ecoscope.platform.annotations import AnyDataFrame

FRAME_CODEC_ID = "ecoscope.frame/v1"
ARROW_CODEC_ID = "ecoscope.arrow/v1"
KEYED_CODEC_ID = "ecoscope.keyed/v1"
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


def encode_keyed_iterable(value: Any) -> bytes:
    """Encode as a JSON header of structure plus a concatenation of leaf blobs.

    The header carries the structure because a `CompositeFilter` is a tuple and
    so neither a JSON object key nor distinguishable from a list once encoded;
    the blobs stay raw so a frame costs its parquet size, not a third more
    as base64.
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
    return _KEYED_MAGIC + struct.pack(">Q", len(header)) + header + b"".join(blobs)


def decode_keyed_iterable(data: bytes) -> Any:
    if not data.startswith(_KEYED_MAGIC):
        raise ValueError("payload is not a keyed iterable")
    start = len(_KEYED_MAGIC)
    (length,) = struct.unpack(">Q", data[start : start + 8])
    body = start + 8
    header = json.loads(data[body : body + length])
    return _unwalk(header, data[body + length :])


def digest_keyed_iterable(value: Any) -> str:
    """Digest structure plus each leaf's own digest — never the encoded bytes.

    Note the key is digested losslessly rather than through `_hash_grouper_key`:
    that encoding drops the operator and truncates to 24 bits, which is a fine
    filename and far too weak to carry cache identity.
    """

    def leaf(codec: Codec, v: Any) -> dict[str, Any]:
        return {"k": "c", "id": codec.codec_id, "d": codec.digest(v)}

    return hashlib.sha256(json.dumps(_walk(value, leaf), sort_keys=True).encode("utf-8")).hexdigest()


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
# refused -- `set_groupers` returns `list[ValueGrouper | ...]`, and a hit would
# hand `split_groups` a list of reprs. This sits between the two and turns that
# into "not cached", which is what refusing was supposed to mean.
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
            digest_keyed_iterable,
            encode_keyed_iterable,
            decode_keyed_iterable,
            matches=is_keyed_iterable,
        ),
        GUARD_CODEC,
    ]
