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
        Field(description="JSON map of table type to data set results root", exclude=True),
    ],
    table_type: Annotated[str, Field(description="The table type to read", exclude=True)],
    optional: Annotated[
        bool,
        Field(description="Return an empty frame if the table type is not bound to a root", exclude=True),
    ] = False,
) -> Annotated[AnyDataFrameOrEmpty | AnyGeoDataFrameOrEmpty, Field()]:
    """Read one typed table from the data set results root bound to its table type.

    Args:
        data_set_roots: JSON map of table type to data set results root.
        table_type: The table type to read.
        optional: Return an empty frame if the table type is absent from the map.

    Returns:
        A GeoDataFrame for spatial table types, otherwise a DataFrame.
    """
    try:
        roots = json.loads(data_set_roots)
    except json.JSONDecodeError:
        roots = None
    if not isinstance(roots, dict):
        raise DataSetSchemaError("ECOSCOPE_WORKFLOWS_DATA_SETS is not a JSON object")

    registered = get_table_type(table_type)

    if table_type not in roots:
        if optional:
            return table_to_frame(registered.schema.empty_table(), table_type)  # type: ignore[return-value]
        raise DataSetSchemaError(f"{table_type}: no data set bound; bound: {', '.join(sorted(roots))}")

    root = roots[table_type]
    manifest = _read_manifest(root, table_type)
    if table_type not in manifest.tables:
        raise DataSetSchemaError(
            f"{table_type}: not in the data set at {root!r}; available: {', '.join(sorted(manifest.tables))}"
        )
    entry = manifest.tables[table_type]
    table = read_table(root, table_type, entry.schema_version, entry.path)
    return table_to_frame(table, table_type)  # type: ignore[return-value]
