from dataclasses import dataclass
from pathlib import Path

import geoarrow.pyarrow as ga  # type: ignore[import-untyped]
import geopandas as gpd  # type: ignore[import-untyped]
import pandas as pd
import pyarrow as pa  # type: ignore[import-untyped]
from pydantic import BaseModel

from ecoscope.platform.annotations import CacheableModel
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


class DataSetManifest(BaseModel, CacheableModel):
    data_set: str
    tables: dict[str, DataSetTable]


_STR = pa.string()
_TS = pa.timestamp("ns", tz="UTC")
_F64 = pa.float64()


def _geometry() -> pa.DataType:
    return ga.wkb()


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


def validate_table(table: pa.Table, table_type: str, schema_version: str | None = None) -> None:
    if table_type not in TABLE_TYPES:
        raise DataSetSchemaError(f"Unknown table type {table_type!r}; known: {sorted(TABLE_TYPES)}")
    registered = TABLE_TYPES[table_type]
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
    if isinstance(df, gpd.GeoDataFrame):
        table = pa.table(df.to_arrow(geometry_encoding="WKB", index=False))
    else:
        table = pa.Table.from_pandas(df, preserve_index=False)
    validate_table(table, table_type)
    return table


def table_to_frame(table: pa.Table, table_type: str) -> pd.DataFrame | gpd.GeoDataFrame:
    if TABLE_TYPES[table_type].spatial:
        return gpd.GeoDataFrame.from_arrow(table)
    return table.to_pandas()


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


def read_table(root_path: str, table_type: str, schema_version: str) -> pa.Table:
    write_path, _ = _get_path(root_path, table_filename(table_type))
    table = pa.ipc.open_file(pa.BufferReader(write_path.read_bytes())).read_all()
    validate_table(table, table_type, schema_version)
    return table
