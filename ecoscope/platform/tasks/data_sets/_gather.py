from pathlib import Path
from typing import Annotated

from pydantic import Field
from wt_registry import register

from ecoscope.platform.annotations import AnyDataFrameOrEmpty, AnyGeoDataFrameOrEmpty
from ecoscope.platform.data_sets import (
    MANIFEST_FILENAME,
    TABLE_TYPES,
    DataSetManifest,
    DataSetSchemaError,
    DataSetTable,
    frame_to_table,
    table_filename,
    write_table,
)
from ecoscope.platform.serde import _get_path, _persist_text


@register(tags=["no-cache"])
def gather_data_sets(
    data_set: Annotated[
        str,
        Field(description="The dataset output key (outputs/<key>.yaml), e.g. patrols", exclude=True),
    ],
    tables: Annotated[
        dict[str, AnyDataFrameOrEmpty | AnyGeoDataFrameOrEmpty],
        Field(description="Table type to frame", exclude=True),
    ],
    root_path: Annotated[str, Field(exclude=True)],
) -> Annotated[DataSetManifest, Field()]:
    """Write each table as a typed Arrow IPC file plus a manifest into the results folder.

    Args:
        data_set: The dataset output key.
        tables: Mapping of table type to the frame holding its rows.
        root_path: The results folder.

    Returns:
        The manifest describing the written tables.
    """
    manifest_path, _ = _get_path(root_path, MANIFEST_FILENAME)
    unknown = [t for t in tables if t not in TABLE_TYPES]
    if unknown:
        raise DataSetSchemaError(f"unknown table type {unknown[0]!r}; known: {sorted(TABLE_TYPES)}")

    converted = {}
    for table_type, df in tables.items():
        if df.empty and len(df.columns) == 0:
            converted[table_type] = TABLE_TYPES[table_type].schema.empty_table()
        else:
            converted[table_type] = frame_to_table(df, table_type)

    entries = {}
    for table_type, table in converted.items():
        _, rows = write_table(table, root_path, table_type)
        entries[table_type] = DataSetTable(
            path=table_filename(table_type),
            rows=rows,
            schema_version=TABLE_TYPES[table_type].schema_version,
        )

    manifest = DataSetManifest(data_set=data_set, tables=entries)
    if isinstance(manifest_path, Path):
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
    _persist_text(manifest.model_dump_json(), root_path, MANIFEST_FILENAME)
    return manifest
