import logging
from typing import Annotated, Literal, cast

from pydantic import BaseModel, Field
from pydantic.json_schema import SkipJsonSchema
from wt_registry import register

from ecoscope.platform.annotations import AdvancedField, AnyDataFrame
from ecoscope.platform.tasks.transformation._unit import Unit, with_unit

logger = logging.getLogger(__name__)


@register()
def map_values(
    df: AnyDataFrame,
    column_name: Annotated[str, Field(description="The column name to map.")],
    value_map: Annotated[dict[str, str], Field(default={}, description="A dictionary of values to map.")],
    missing_values: Annotated[
        Literal["preserve", "remove", "replace"],
        Field(
            default="remove",
            description="How to handle values that aren't in value_map.",
        ),
    ],
    replacement: Annotated[
        str | SkipJsonSchema[None],
        Field(default=None, description="The replacement for values not in value_map."),
    ] = None,
) -> AnyDataFrame:
    match missing_values:
        case "preserve":
            df[column_name] = df[column_name].map(value_map).fillna(df[column_name])
        case "remove":
            df[column_name] = df[column_name].map(value_map)
        case "replace":
            if replacement is None:
                raise ValueError("replacement param must be provided if missing_values is 'replace'")
            df[column_name] = df[column_name].map(value_map).fillna(replacement)
        case _:
            raise ValueError("Invalid selection for missing_values")

    return cast(AnyDataFrame, df)


@register()
def assign_value(
    df: AnyDataFrame,
    column_name: Annotated[str, Field(description="The column name to map.")],
    value: Annotated[
        str | int | float | bool | SkipJsonSchema[None],
        Field(description="The column value."),
    ],
    noop_if_column_exists: Annotated[
        bool,
        Field(
            description="If set to true and column_name exists on df, do nothing",
            default=False,
        ),
    ] = False,
) -> AnyDataFrame:
    if not noop_if_column_exists or column_name not in df.columns:
        df[column_name] = value
    return cast(AnyDataFrame, df)


@register()
def map_values_with_unit(
    df: AnyDataFrame,
    input_column_name: Annotated[str, Field(description="The column name to map.")],
    output_column_name: Annotated[str, Field(description="The new column name.")],
    original_unit: Annotated[
        Unit | SkipJsonSchema[None],
        Field(description="The original unit of measurement."),
    ] = None,
    new_unit: Annotated[
        Unit | SkipJsonSchema[None],
        Field(description="The unit to convert to."),
    ] = None,
    decimal_places: Annotated[
        int,
        AdvancedField(default=1, description="The number of decimal places to display."),
    ] = 1,
) -> AnyDataFrame:
    def format_with_unit(x):
        data = with_unit(x, original_unit=original_unit, new_unit=new_unit)
        return f"{data.value:.{decimal_places}f} {data.unit or ''}".strip()

    df[output_column_name] = df[input_column_name].apply(format_with_unit)
    return df


class RenameColumn(BaseModel):
    original_name: str
    new_name: str


@register()
def map_columns(
    df: AnyDataFrame,
    drop_columns: Annotated[
        list[str] | SkipJsonSchema[None],
        AdvancedField(default=[], description="List of columns to drop."),
    ] = None,
    retain_columns: Annotated[
        list[str] | SkipJsonSchema[None],
        AdvancedField(
            default=[],
            description="""List of columns to retain with the order specified by the list.
                        Keep all the columns if the list is empty.""",
        ),
    ] = None,
    rename_columns: Annotated[
        list[RenameColumn] | SkipJsonSchema[dict[str, str]] | SkipJsonSchema[None],
        AdvancedField(default={}, description="Dictionary of columns to rename."),
    ] = None,
    raise_if_not_found: Annotated[
        bool, Field(description="Whether or not to raise if var is not in value_map.")
    ] = True,
) -> AnyDataFrame:
    """
    Maps and transforms the columns of a DataFrame based on the provided parameters. The order of the operations is as
    follows: drop columns, retain/reorder columns, and rename columns.

    Args:
        df (AnyDataFrame): The input DataFrame to be transformed.
        drop_columns (list[str]): List of columns to drop from the DataFrame.
        retain_columns (list[str]): List of columns to retain. The order of columns will be preserved.
        rename_columns (dict[str, str]): Dictionary of columns to rename.
        raise_if_not_found (bool): Whether or not to raise in the event a column is not found.

    Returns:
        AnyDataFrame: The transformed DataFrame.

    Raises:
        KeyError: If any of the columns specified are not found in the DataFrame.
    """

    if drop_columns:
        if "geometry" in drop_columns:
            logger.warning("'geometry' found in drop_columns, which may affect spatial operations.")
        df = df.drop(
            columns=drop_columns,
            errors="ignore" if not raise_if_not_found else "raise",
        )

    if retain_columns:
        if raise_if_not_found and any(col not in df.columns for col in retain_columns):
            raise KeyError(f"Columns {retain_columns} not all found in DataFrame.")
        df = df.reindex(columns=retain_columns)  # type: ignore[assignment]

    if rename_columns:
        if isinstance(rename_columns, list):
            rename_columns = {item.original_name: item.new_name for item in rename_columns}

        if "geometry" in rename_columns.keys():
            logger.warning("'geometry' found in rename_columns, which may affect spatial operations.")
        if raise_if_not_found and any(col not in df.columns for col in rename_columns.keys()):
            raise KeyError(
                f"Columns {list(rename_columns.keys())} not all found in DataFrame. Existing columns: {df.columns}"
            )
        df = df.rename(columns=rename_columns)  # type: ignore[assignment]

    return cast(AnyDataFrame, df)


@register()
def title_case_columns_by_prefix(
    df: AnyDataFrame,
    prefix: Annotated[
        str,
        Field(description="Column names prefixed with this value will be converted to title case."),
    ],
) -> AnyDataFrame:
    """
    Convert the column names beginning with the provided prefix to title case.

    Args:
        df (AnyDataFrame): The input DataFrame.
        prefix (str): Column names prefixed with this value will be converted to title case.

    Returns:
        AnyDataFrame: The updated DataFrame.
    """

    mapping = {col: col.removeprefix(prefix).replace("_", " ").title() for col in df.columns if col.startswith(prefix)}
    df = df.rename(columns=mapping)  # type: ignore[assignment]

    return cast(AnyDataFrame, df)


@register()
def reorder_columns(
    df: AnyDataFrame,
    columns: Annotated[
        list[str],
        Field(description="Provided column names will be first in the dataframe."),
    ],
) -> AnyDataFrame:
    """
    Reorder columns in the provided dataframe to the order of the provided column names.

    Args:
        df (AnyDataFrame): The input DataFrame.
        columns (list[str]): Provided column names will be first in the dataframe.

    Returns:
        AnyDataFrame: The updated DataFrame.
    """
    assert all([col in df for col in columns])

    reorderd = columns + [col for col in df.columns if col not in columns]

    df = df.reindex(columns=reorderd)

    return cast(AnyDataFrame, df)


@register()
def fill_na(
    df: AnyDataFrame,
    value: Annotated[
        str | int | float | bool | SkipJsonSchema[None],
        Field(description="The value to fill."),
    ],
    columns: Annotated[
        list[str] | SkipJsonSchema[None],
        Field(description="Provided columns will have nan values filled."),
    ] = None,
) -> AnyDataFrame:
    """
    Fill NA values the with the input value.

    Args:
        df (AnyDataFrame): The input DataFrame.
        value (str | int | float | bool | None): The value to fill NaN with.
        columns (list[str]): If provided, fill these column only.

    Returns:
        AnyDataFrame: The updated DataFrame.
    """
    df = df.fillna(value) if columns is None else df.fillna({col: value for col in columns})
    return cast(AnyDataFrame, df)


@register()
def strip_prefix_from_column_names(
    df: AnyDataFrame,
    prefix: Annotated[
        str,
        Field(description="The prefix to remove."),
    ],
) -> AnyDataFrame:
    """
    Strip the provided prefix from column names that have it.

    Args:
        df (AnyDataFrame): The input DataFrame.
        prefix (str): The prefix to remove from column names in this dataframe.

    Returns:
        AnyDataFrame: The updated DataFrame.
    """
    df = df.rename(columns={col: col.removeprefix(prefix) for col in df.columns})  # type: ignore[assignment]
    return cast(AnyDataFrame, df)


@register()
def lookup_string_var(
    var: Annotated[str, Field(...)],
    value_map: Annotated[dict[str, str], Field(default={}, description="A dictionary of values.")],
    raise_if_not_found: Annotated[
        bool, Field(description="Whether or not to raise if var is not in value_map.")
    ] = True,
) -> str:
    """
    Lookup `var` in `value_map` and return the string mapped by `var`
    If `raise_if_not_found` is true, raises `KeyError` if `var` is not in `value_map`
    If `raise_if_not_found` is false, `var` is passed through unchanged

    Args:
        var (str): The input var.
        value_map (dict[str, str]): The map to lookup `var` in.
        raise_if_not_found (bool): Whether or not to raise in the event `var` is not found.

    Returns:
        str: The mapped value, or `var`.
    Raises:
        KeyError: If  `var` is not found in `value_map`.
    """
    if raise_if_not_found:
        return value_map[var]
    else:
        return value_map.get(var, var)
