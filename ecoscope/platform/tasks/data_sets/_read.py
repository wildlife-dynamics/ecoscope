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
        Field(description="JSON map of binding name to the table bound to it", exclude=True),
    ],
    table_type: Annotated[str, Field(description="The table type this step needs", exclude=True)],
    binding: Annotated[
        str | None,
        Field(description="The binding's name in the map; defaults to the table type", exclude=True),
    ] = None,
) -> Annotated[AnyDataFrameOrEmpty | AnyGeoDataFrameOrEmpty, Field()]:
    """Read the table bound to this step.

    A workflow says only which type of table it needs. Which table of which data set
    feeds it is decided at run time, in the map: per binding name, `{"url": the data
    set's results root, "table_id": the table's id there}`, or a bare root when the
    table's id is the binding's name. The binding is named after the table type unless
    a workflow reads two tables of one type and names them apart.

    Args:
        data_set_roots: JSON map of binding names to bound tables.
        table_type: The type the bound table must have.
        binding: The binding's name in the map. Defaults to `table_type`.

    Returns:
        A GeoDataFrame for spatial table types, otherwise a DataFrame.
    """
    try:
        roots = json.loads(data_set_roots)
    except json.JSONDecodeError:
        roots = None
    if not isinstance(roots, dict):
        raise DataSetSchemaError("ECOSCOPE_WORKFLOWS_DATA_SETS is not a JSON object")

    get_table_type(table_type)
    name = binding or table_type

    if name not in roots:
        raise DataSetSchemaError(f"{name}: no data set bound; bound: {', '.join(sorted(roots))}")

    bound = roots[name]
    root, table_id = (bound.get("url"), bound.get("table_id", name)) if isinstance(bound, dict) else (bound, name)
    if not isinstance(root, str) or not isinstance(table_id, str):
        raise DataSetSchemaError(f"{name}: a binding is a results root or {{url, table_id}}, not {bound!r}")
    manifest = _read_manifest(root, name)
    if table_id not in manifest.tables:
        raise DataSetSchemaError(
            f"{name}: no table {table_id!r} in the data set at {root!r}; "
            f"available: {', '.join(sorted(manifest.tables))}"
        )
    entry = manifest.tables[table_id]
    if entry.table_type != table_type:
        raise DataSetSchemaError(
            f"{name}: expected a {table_type} table, but {table_id!r} in the data set at {root!r} "
            f"is a {entry.table_type} table"
        )
    table = read_table(root, table_type, entry.schema_version, entry.path)
    return table_to_frame(table, table_type)  # type: ignore[return-value]
