import json
from typing import Annotated

from pydantic import Field, ValidationError
from wt_registry import register

from ecoscope.platform.annotations import AnyDataFrameOrEmpty, AnyGeoDataFrameOrEmpty
from ecoscope.platform.data_sets import (
    MANIFEST_FILENAME,
    DataSetManifest,
    DataSetSchemaError,
    get_table_type,
    read_table,
    table_to_frame,
)
from ecoscope.platform.serde import _get_path


def _read_manifest(root: str, table_type: str | None = None) -> DataSetManifest:
    path, _ = _get_path(root, MANIFEST_FILENAME)
    subject = f"{table_type}: " if table_type else ""
    try:
        text = path.read_text()
    except FileNotFoundError as e:
        raise DataSetSchemaError(f"{subject}no data set manifest found at {root!r}") from e
    try:
        return DataSetManifest.model_validate_json(text)
    except ValidationError as e:
        raise DataSetSchemaError(f"{subject}invalid data set manifest at {root!r}: {e}") from e


@register()
def read_data_set_manifest(
    data_set_root: Annotated[str, Field(description="The data set results root", exclude=True)],
) -> Annotated[DataSetManifest, Field()]:
    """Read the manifest of a data set results folder.

    Args:
        data_set_root: The results folder of the data set workflow.

    Returns:
        The manifest describing the data set's tables.
    """
    return _read_manifest(data_set_root)


@register()
def read_data_set_table(
    data_set_roots: Annotated[
        str,
        Field(description="JSON map of table id to where the bound table is", exclude=True),
    ],
    table_id: Annotated[str, Field(description="What this workflow calls the table it reads", exclude=True)],
    table_type: Annotated[
        str | None,
        Field(description="The table type this workflow expects; defaults to the table id", exclude=True),
    ] = None,
) -> Annotated[AnyDataFrameOrEmpty | AnyGeoDataFrameOrEmpty, Field()]:
    """Read the typed table bound to one of this workflow's table ids.

    A workflow names each table it reads and says which type it must be. What it is
    bound to is matched by type, not by name: the map gives, per table id, either the
    results root of a data set whose table has the same id, or `{"url": root,
    "table_id": id}` naming another table of that data set.

    Args:
        data_set_roots: JSON map of this workflow's table ids to bound tables.
        table_id: The table to read, as this workflow names it.
        table_type: The type the table must have. Defaults to `table_id`.

    Returns:
        A GeoDataFrame for spatial table types, otherwise a DataFrame.
    """
    try:
        roots = json.loads(data_set_roots)
    except json.JSONDecodeError:
        roots = None
    if not isinstance(roots, dict):
        raise DataSetSchemaError("ECOSCOPE_WORKFLOWS_DATA_SETS is not a JSON object")

    expected = table_type or table_id
    get_table_type(expected)

    if table_id not in roots:
        raise DataSetSchemaError(f"{table_id}: no data set bound; bound: {', '.join(sorted(roots))}")

    bound = roots[table_id]
    root, source_id = (bound["url"], bound.get("table_id", table_id)) if isinstance(bound, dict) else (bound, table_id)
    manifest = _read_manifest(root, table_id)
    if source_id not in manifest.tables:
        raise DataSetSchemaError(
            f"{table_id}: no table {source_id!r} in the data set at {root!r}; "
            f"available: {', '.join(sorted(manifest.tables))}"
        )
    entry = manifest.tables[source_id]
    if entry.table_type != expected:
        raise DataSetSchemaError(
            f"{table_id}: expected a {expected} table, but {source_id!r} in the data set at {root!r} "
            f"is a {entry.table_type} table"
        )
    table = read_table(root, expected, entry.schema_version, entry.path)
    return table_to_frame(table, expected)  # type: ignore[return-value]
