"""How an `ecoscope.platform` task return becomes bytes, and back again.

Only leaves live here. Which codec applies is declared by the task's return
type -- `CodecRef` on a type alias, `__wt_codec__` on a class -- resolved once
when wt-registry serializes the library, and assembled into one `wt_task.cache.Tree`
per step at run time. Nothing in this module inspects a value to decide what it is.

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
from typing import TYPE_CHECKING, Any

import pandas as pd
import pyarrow as pa
from wt_task.cache import SKIP, Codec, Tree, Undigestible

if TYPE_CHECKING:
    from pydantic import BaseModel

    from ecoscope.platform.annotations import AnyDataFrame

FRAME_CODEC_ID = "ecoscope.frame/v1"
ARROW_CODEC_ID = "ecoscope.arrow/v1"
KEYED_CODEC_ID = "ecoscope.keyed/v1"
GROUPER_CODEC_ID = "ecoscope.grouper/v1"
MODEL_CODEC_ID = "ecoscope.model/v1"


class UnencodableValue(Undigestible):
    """Raised by a leaf handed a value it cannot carry losslessly."""


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


def _cell_to_json(value: Any) -> dict[str, Any]:
    """Describe a structural cell as a tree JSON can carry.

    Structure is spelled out rather than left to `json.dumps` because a `list`
    and a `tuple` cell are not the same value, and nothing else in the frame
    records which one a column held.
    """
    if value is None or isinstance(value, (str, bool, int, float)):
        return {"k": "j", "v": value}
    if isinstance(value, (list, tuple)):
        return {"k": "list" if isinstance(value, list) else "tuple", "v": [_cell_to_json(v) for v in value]}
    if isinstance(value, dict):
        if not all(isinstance(k, str) for k in value):
            raise UnencodableValue(f"dict with non-string keys: {list(value)[:3]!r}")
        return {"k": "dict", "v": {k: _cell_to_json(v) for k, v in value.items()}}
    raise UnencodableValue(f"{type(value).__name__} cannot be stored inside a dataframe cell")


def _cell_from_json(node: dict[str, Any]) -> Any:
    kind = node["k"]
    if kind == "j":
        return node["v"]
    if kind == "list":
        return [_cell_from_json(n) for n in node["v"]]
    if kind == "tuple":
        return tuple(_cell_from_json(n) for n in node["v"])
    return {k: _cell_from_json(n) for k, n in node["v"].items()}


def _encode_cell(value: Any) -> str:
    return json.dumps(_cell_to_json(value))


def _decode_cell(text: str) -> Any:
    return _cell_from_json(json.loads(text))


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
# why this is not `exact`. The two frame formats share no content-id space,
# because ids are namespaced by codec, so the same data as a frame and as a
# table digests differently. Cross-format dedup is the price of not having to
# agree on a canonical form across two type systems.
FRAME = Codec(
    codec_id=FRAME_CODEC_ID,
    media_type="application/vnd.apache.parquet",
    types=(pd.DataFrame,),
    encode=encode_dataframe,
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

    `exact`, and keyed by class in the codec id, because the payload no longer
    names the class to rebuild: a subclass encoded through its parent's codec
    would come back narrowed, and a declaration that moved has to miss rather
    than decode as something else.
    """
    return Codec(
        codec_id=f"{MODEL_CODEC_ID}/{cls.__module__}.{cls.__qualname__}",
        media_type="application/json",
        types=(cls,),
        encode=_model_json,
        decode=cls.model_validate_json,
        digest=lambda v: hashlib.sha256(_model_json(v)).hexdigest(),
        exact=True,
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
        exact=True,
    )
