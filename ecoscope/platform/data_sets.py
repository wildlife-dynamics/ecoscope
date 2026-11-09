from dataclasses import dataclass

import geoarrow.pyarrow as ga  # type: ignore[import-untyped]
import pyarrow as pa  # type: ignore[import-untyped]
from pydantic import BaseModel

from ecoscope.platform.annotations import CacheableModel

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
                ("patrol_start_time", _STR),
                ("patrol_end_time", _STR),
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
