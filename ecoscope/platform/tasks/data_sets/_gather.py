from pathlib import Path
from typing import Annotated, Any

import pandas as pd
from pydantic import BeforeValidator, Field
from typing_extensions import TypedDict
from wt_registry import register
from wt_task.skip import SkippedDependencyFallback, SkipSentinel

from ecoscope.platform.annotations import AnyDataFrameOrEmpty, AnyGeoDataFrameOrEmpty
from ecoscope.platform.data_sets import (
    MANIFEST_FILENAME,
    TABLE_TYPES,
    DataSetManifest,
    DataSetTable,
    frame_to_table,
    get_table_type,
    table_filename,
    write_table,
)
from ecoscope.platform.serde import _get_path, _persist_text


def _skipped_to_empty(df: Any) -> Any:
    return pd.DataFrame() if isinstance(df, SkipSentinel) else df


class GatherTable(TypedDict):
    """One table of a data set: its registered type and the frame holding its rows."""

    table_type: str
    # A skipped step is written as an empty table. Both spellings: the task framework
    # reads the first, plain validation of the nested value the second.
    frame: Annotated[
        AnyDataFrameOrEmpty | AnyGeoDataFrameOrEmpty,
        SkippedDependencyFallback(_skipped_to_empty),
        BeforeValidator(_skipped_to_empty),
    ]


# Tagged `no-cache`: the task exists for the files it writes into this run's results
# folder. A hit would return the manifest without writing them, and the manifest
# would name tables that are not there.
@register(tags=["no-cache"])
def gather_data_sets(
    data_set: Annotated[
        str,
        Field(description="The dataset output key (outputs/<key>.yaml), e.g. patrols", exclude=True),
    ],
    tables: Annotated[
        dict[str, GatherTable],
        Field(description="Table id to its table type and frame", exclude=True),
    ],
    root_path: Annotated[str, Field(exclude=True)],
) -> Annotated[DataSetManifest, Field()]:
    """Write each table as a typed Arrow IPC file plus a manifest into the results folder.

    Args:
        data_set: The dataset output key.
        tables: Mapping of table id to `{table_type, frame}`. Two tables may share a
            type; where a data set has one table of a type, name it after the type.
        root_path: The results folder.

    Returns:
        The manifest describing the written tables.
    """
    manifest_path, _ = _get_path(root_path, MANIFEST_FILENAME)
    for table_id, entry in tables.items():
        table_filename(table_id)
        get_table_type(entry["table_type"])

    converted = {}
    for table_id, entry in tables.items():
        table_type, df = entry["table_type"], entry["frame"]
        if df.empty and len(df.columns) == 0:
            converted[table_id] = TABLE_TYPES[table_type].schema.empty_table()
        else:
            converted[table_id] = frame_to_table(df, table_type)

    entries = {}
    for table_id, table in converted.items():
        _, rows = write_table(table, root_path, table_id)
        table_type = tables[table_id]["table_type"]
        entries[table_id] = DataSetTable(
            table_type=table_type,
            path=table_filename(table_id),
            rows=rows,
            schema_version=TABLE_TYPES[table_type].schema_version,
        )

    manifest = DataSetManifest(data_set=data_set, tables=entries)
    if isinstance(manifest_path, Path):
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
    _persist_text(manifest.model_dump_json(), root_path, MANIFEST_FILENAME)
    return manifest
