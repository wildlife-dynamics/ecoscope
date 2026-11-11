import json
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import geoarrow.pyarrow as ga  # type: ignore[import-untyped]
import geopandas as gpd  # type: ignore[import-untyped]
import pandas as pd
import pyarrow as pa  # type: ignore[import-untyped]
from pydantic import BaseModel

from ecoscope.platform.annotations import CacheableModel
from ecoscope.platform.codecs import (
    TREE_COLUMNS_KEY,
    UnencodableValue,
    decode_cell,
    encode_cell,
    object_series,
    tree_columns,
)
from ecoscope.platform.serde import _get_path, _persist_bytes

MANIFEST_FILENAME = "data_sets/manifest.json"


def table_filename(table_type: str) -> str:
    return f"data_sets/{table_type}.arrow"


class DataSetSchemaError(ValueError):
    pass


@dataclass(frozen=True)
class TableType:
    name: str
    schema_version: str
    schema: pa.Schema
    spatial: bool


class DataSetTable(BaseModel):
    path: str
    rows: int
    schema_version: str


# A key of `TABLE_TYPES`.
TableTypeName = str


class DataSetManifest(BaseModel, CacheableModel):
    """What a data set run wrote: one entry per table, keyed by table type name.

    The `TableType` itself is not here: its Arrow schema lives in `TABLE_TYPES`, the
    one definition writer and reader both import, and `schema_version` says which
    version of it the file was written with. Keys are not checked against the
    registry, so a reader can take its table from a manifest that also lists a type
    it does not know yet.
    """

    data_set: str
    tables: dict[TableTypeName, DataSetTable]


_STR = pa.string()
_TS = pa.timestamp("ns", tz="UTC")
_F64 = pa.float64()


def _geometry() -> pa.DataType:
    return ga.wkb().with_crs("EPSG:4326")


def _table_type(name: str, fields: list[tuple[str, pa.DataType]], spatial: bool) -> TableType:
    return TableType(name=name, schema_version="1", schema=pa.schema(fields), spatial=spatial)


_EVENT_FIELDS: list[tuple[str, pa.DataType]] = [
    ("event_type", _STR),
    ("time", _TS),
    ("geometry", _geometry()),
]

TABLE_TYPES: dict[str, TableType] = {
    t.name: t
    for t in [
        _table_type(
            "patrol_tracks",
            [
                ("groupby_col", _STR),
                ("segment_start", _TS),
                ("segment_end", _TS),
                ("timespan_seconds", _F64),
                ("dist_meters", _F64),
                ("speed_kmhr", _F64),
                ("heading", _F64),
                ("junk_status", pa.bool_()),
                ("geometry", _geometry()),
                ("patrol_id", _STR),
                ("patrol_serial_number", _STR),
                ("patrol_type", _STR),
            ],
            spatial=True,
        ),
        _table_type("patrol_events", [*_EVENT_FIELDS, ("patrol_id", _STR)], spatial=True),
        _table_type(
            "patrol_legs",
            [
                ("patrol_id", _STR),
                ("patrol_serial_number", _STR),
                ("patrol_segment_id", _STR),
                ("patrol_type", _STR),
                ("patrol_subject", _STR),
                ("segment_start", _TS),
                ("segment_end", _TS),
                ("leg_active_seconds", _F64),
                ("leg_member_days", _F64),
                ("patrol_elapsed_seconds", _F64),
            ],
            spatial=False,
        ),
        _table_type(
            "spatial_feature_groups",
            [
                ("pk", _STR),
                ("name", _STR),
                ("short_name", _STR),
                ("feature_type", _STR),
                ("metadata", pa.struct([("id", _STR), ("display_name", _STR)])),
                ("geometry", _geometry()),
            ],
            spatial=True,
        ),
        _table_type("events", _EVENT_FIELDS, spatial=True),
    ]
}


def _is_compatible(expected: pa.DataType, found: pa.DataType) -> bool:
    if pa.types.is_null(found):
        return True
    if isinstance(expected, pa.ExtensionType):
        return isinstance(found, pa.ExtensionType) and found.extension_name.startswith("geoarrow.")
    if pa.types.is_timestamp(expected):
        return pa.types.is_timestamp(found) and found.tz is not None
    if pa.types.is_string(expected):
        return pa.types.is_string(found) or pa.types.is_large_string(found)
    if pa.types.is_struct(expected):
        return pa.types.is_struct(found) and all(
            found.get_field_index(f.name) >= 0 and _is_compatible(f.type, found.field(f.name).type) for f in expected
        )
    return expected == found


def get_table_type(table_type: str) -> TableType:
    if table_type not in TABLE_TYPES:
        raise DataSetSchemaError(f"unknown table type {table_type!r}; known: {sorted(TABLE_TYPES)}")
    return TABLE_TYPES[table_type]


def validate_table(table: pa.Table, table_type: str, schema_version: str | None = None) -> None:
    registered = get_table_type(table_type)
    if schema_version is not None and schema_version != registered.schema_version:
        raise DataSetSchemaError(
            f"{table_type}: schema_version mismatch, expected {registered.schema_version}, found {schema_version}"
        )

    missing = []
    mistyped = []
    for field in registered.schema:
        idx = table.schema.get_field_index(field.name)
        if idx < 0:
            missing.append(field.name)
        elif not _is_compatible(field.type, table.schema.field(idx).type):
            mistyped.append(f"{field.name} (expected {field.type}, found {table.schema.field(idx).type})")

    if missing or mistyped:
        problems = []
        if missing:
            problems.append(f"missing fields: {', '.join(missing)}")
        if mistyped:
            problems.append(f"mistyped fields: {', '.join(mistyped)}")
        raise DataSetSchemaError(
            f"{table_type} (schema_version {registered.schema_version}) failed validation; " + "; ".join(problems)
        )


def frame_to_table(df: pd.DataFrame | gpd.GeoDataFrame, table_type: str) -> pa.Table:
    """Convert a frame to the Arrow table of its type, index dropped.

    Columns beyond the registered schema that hold a `list`, `tuple` or `dict`, or
    mix scalar types, are stored as JSON text the way the step cache stores them
    (`codecs.encode_cell`), and named in the schema metadata so `table_to_frame`
    restores them. Arrow would otherwise infer a struct per column, which fails on
    conflicting value types and reads back as something other than what was written.
    """
    registered = get_table_type(table_type)
    trees = [name for name in tree_columns(df) if name not in registered.schema.names]  # type: ignore[arg-type]
    if trees:
        df = df.copy()
        for name in trees:
            try:
                df[name] = [encode_cell(v) for v in df[name]]
            except UnencodableValue as e:
                raise DataSetSchemaError(f"{table_type}: cannot store column {name!r}: {e}") from e
    try:
        if isinstance(df, gpd.GeoDataFrame):
            table = pa.table(df.to_arrow(geometry_encoding="WKB", index=False))
        else:
            table = pa.Table.from_pandas(df, preserve_index=False)
    except pa.ArrowException as e:
        match = re.search(r"Conversion failed for column (\S+)", str(e))
        column = f" column {match.group(1)!r}" if match else ""
        raise DataSetSchemaError(f"{table_type}: cannot convert{column} to Arrow: {e}") from e
    if trees:
        table = table.replace_schema_metadata(
            {**(table.schema.metadata or {}), TREE_COLUMNS_KEY: json.dumps(trees).encode()}
        )
    validate_table(table, table_type)
    return table


def table_to_frame(table: pa.Table, table_type: str) -> pd.DataFrame | gpd.GeoDataFrame:
    df = gpd.GeoDataFrame.from_arrow(table) if TABLE_TYPES[table_type].spatial else table.to_pandas()
    if (raw := (table.schema.metadata or {}).get(TREE_COLUMNS_KEY)) is not None:
        for name in json.loads(raw):
            df[name] = object_series([decode_cell(v) for v in df[name]], df.index)
    return df


def write_table(table: pa.Table, root_path: str, table_type: str) -> tuple[str, int]:
    filename = table_filename(table_type)
    sink = pa.BufferOutputStream()
    with pa.ipc.new_file(sink, table.schema) as writer:
        writer.write_table(table)
    write_path, _ = _get_path(root_path, filename)
    if isinstance(write_path, Path):
        write_path.parent.mkdir(parents=True, exist_ok=True)
    read_path = _persist_bytes(sink.getvalue().to_pybytes(), root_path, filename)
    return read_path, table.num_rows


def read_table(root_path: str, table_type: str, schema_version: str, path: str | None = None) -> pa.Table:
    """Read and validate one table; `path` is its manifest entry's, relative to the root."""
    filename = table_filename(table_type) if path is None else path
    if PurePosixPath(filename).is_absolute() or ".." in PurePosixPath(filename).parts:
        raise DataSetSchemaError(f"{table_type}: table path {filename!r} leaves the data set root")
    write_path, _ = _get_path(root_path, filename)
    try:
        data = write_path.read_bytes()
    except FileNotFoundError as e:
        raise DataSetSchemaError(f"{table_type}: no table file {filename} at {root_path!r}") from e
    table = pa.ipc.open_file(pa.BufferReader(data)).read_all()
    validate_table(table, table_type, schema_version)
    return table
