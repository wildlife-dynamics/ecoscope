import logging
from collections.abc import Collection
from typing import Annotated, Literal, cast, get_args

from pydantic import BaseModel, Field
from pydantic.json_schema import SkipJsonSchema
from wt_registry import register

from ecoscope.platform.annotations import AdvancedField, AnyDataFrame
from ecoscope.platform.tasks.transformation._unit import Unit, is_linear_unit_conversion, with_unit

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
            df[column_name] = df[column_name].map(value_map).fillna(df7[column_name])
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
    if new_unit is None or original_unit == new_unit:
        # no conversion: just format with the original (or absent) unit
        suffix = f" {original_unit}".rstrip() if original_unit else ""
        values = df[input_column_name].to_numpy()
    elif is_linear_unit_conversion(original_unit, new_unit):
        # multiplicative conversion: probe the factor once and broadcast
        quantity = with_unit(1.0, original_unit=original_unit, new_unit=new_unit)
        suffix = f" {quantity.unit}".rstrip() if quantity.unit else ""
        values = df[input_column_name].to_numpy() * quantity.value
    else:
        # non-linear units ie. dB
        def format_row(x):
            data = with_unit(x, original_unit=original_unit, new_unit=new_unit)
            return f"{data.value:.{decimal_places}f} {data.unit or ''}".strip()

        df[output_column_name] = df[input_column_name].apply(format_row)
        return df

    df[output_column_name] = [f"{v:.{decimal_places}f}{suffix}" for v in values]
    return df


RenameDuplicateStrategy = Literal["overwrite", "skip", "error"]


def _active_geometry_name(df: AnyDataFrame) -> str | None:
    """
    The name of the frame's active geometry column, or None if it has none.

    Args:
        df (AnyDataFrame): The DataFrame to inspect.

    Returns:
        str | None: The active geometry column name.
    """
    import geopandas as gpd  # type: ignore[import-untyped]

    return df.active_geometry_name if isinstance(df, gpd.GeoDataFrame) else None


def _find_rename_collisions(columns: Collection[str], mapping: dict[str, str]) -> dict[str, str]:
    """
    Find the renames in `mapping` whose new name is already taken, either by a column that
    isn't being renamed or by an earlier rename in `mapping`.

    Args:
        columns (Collection[str]): The current column names.
        mapping (dict[str, str]): Mapping of existing column name to new column name.

    Returns:
        dict[str, str]: The colliding subset of `mapping`.
    """
    # The names that will still be occupied once the renames are applied, so that a swap
    # (eg. {"A": "B", "B": "A"}) is not a collision.
    taken = {col for col in columns if col not in mapping}

    collisions: dict[str, str] = {}
    for old, new in mapping.items():
        if new in taken:
            collisions[old] = new
        else:
            taken.add(new)

    return collisions


def _safe_rename_columns(
    df: AnyDataFrame,
    rename_columns: dict[str, str],
    duplicate_strategy: RenameDuplicateStrategy = "skip",
) -> AnyDataFrame:
    """
    Rename columns, resolving new names that are already taken so that the renames never introduce
    duplicate column labels.

    `DataFrame.rename` will happily rename a column to a name that already exists, leaving the
    frame with duplicate labels. `df[name]` then returns a DataFrame rather than a Series, which
    tends to fail confusingly, far from the rename that caused it.

    Duplicate labels among the columns that aren't being renamed are left alone; renaming a column
    whose label is duplicated is an error, since it would carry the duplication over to the new
    name. Use `drop_duplicate_columns` to resolve those first.

    Args:
        df (AnyDataFrame): The DataFrame to rename columns on.
        rename_columns (dict[str, str]): Mapping of existing column name to new column name.
        duplicate_strategy (RenameDuplicateStrategy): How to resolve a new name that is already
            taken, either by a column that isn't being renamed or by an earlier entry in
            `rename_columns`:
            - "skip": skip the rename, leaving the column under its original name
            - "overwrite": drop the column holding the name so the renamed one takes it
            - "error": raise ValueError

    Returns:
        AnyDataFrame: The DataFrame with columns renamed.

    Raises:
        ValueError: If a column being renamed has a duplicated label, if `duplicate_strategy` is
            "error" and a new name is already taken, or if overwriting would drop the active
            geometry column.
    """
    if duplicate_strategy not in get_args(RenameDuplicateStrategy):
        raise ValueError(f"Invalid selection for duplicate_strategy: {duplicate_strategy}")

    # Renames of absent columns are no-ops for pandas, so they can't collide, and identity
    # renames leave the column exactly where it is.
    mapping = {old: new for old, new in rename_columns.items() if old != new and old in df.columns}
    if not mapping:
        return df

    # Reject attempts to rename already duplicated column names
    if duplicated := sorted({col for col in df.columns[df.columns.duplicated()] if col in mapping}):
        raise ValueError(
            f"Cannot rename columns {duplicated} because the DataFrame has more than one column "
            "with each of those names. Resolve them first, eg. with `drop_duplicate_columns`."
        )

    if duplicate_strategy == "error":
        if collisions := _find_rename_collisions(df.columns, mapping):
            raise ValueError(
                f"Renaming columns {mapping} would create duplicate columns: "
                f"{sorted(set(collisions.values()))}. Existing columns: {list(df.columns)}"
            )
        return cast(AnyDataFrame, df.rename(columns=mapping))

    if duplicate_strategy == "skip":
        # Skipping a rename leaves its original name occupied, which can collide with a rename
        # we had already accepted, so keep resolving until nothing collides.
        skipped: dict[str, str] = {}
        while collisions := _find_rename_collisions(df.columns, mapping):
            skipped |= collisions
            mapping = {old: new for old, new in mapping.items() if old not in collisions}
        if skipped:
            logger.warning(f"Not renaming columns whose new name is already taken: {skipped}")
        return cast(AnyDataFrame, df.rename(columns=mapping))

    # "overwrite": track which column will hold each name, so that the last rename to a given
    # name wins and whatever held it - an untouched column, or a column renamed earlier - goes.
    owner: dict[str, str] = {col: col for col in df.columns if col not in mapping}
    resolved: dict[str, str] = {}
    to_drop: list[str] = []
    for old, new in mapping.items():
        if new in owner:
            displaced = owner.pop(new)
            to_drop.append(displaced)
            resolved.pop(displaced, None)
        resolved[old] = new
        owner[new] = old

    if to_drop:
        # Dropping the active geometry column leaves a GeoDataFrame with no geometry, which
        # demotes it to a plain DataFrame and loses the CRS.
        if (active_geometry := _active_geometry_name(df)) in to_drop:
            raise ValueError(
                f"Renaming columns {mapping} would overwrite the active geometry column "
                f"'{active_geometry}'. Drop or rename it explicitly first, or use a "
                "duplicate_strategy of 'skip' or 'error'."
            )
        if "geometry" in to_drop:
            logger.warning("'geometry' is being overwritten by a rename, which may affect spatial operations.")
        logger.warning(f"Overwriting existing columns: {to_drop}")
        df = cast(AnyDataFrame, df.drop(columns=to_drop))

    return cast(AnyDataFrame, df.rename(columns=resolved))


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
    duplicate_strategy: Annotated[
        RenameDuplicateStrategy,
        AdvancedField(
            default="skip",
            description=(
                "Strategy for handling a rename whose new name is already taken. "
                "'skip': skip the rename, leaving the column under its original name; "
                "'overwrite': replace the existing column; "
                "'error': raise ValueError."
            ),
        ),
    ] = "skip",
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
        duplicate_strategy (RenameDuplicateStrategy): How to handle renaming a column to a name that is
            already taken; see `_safe_rename_columns`.

    Returns:
        AnyDataFrame: The transformed DataFrame.

    Raises:
        KeyError: If any of the columns specified are not found in the DataFrame.
        ValueError: If a column being renamed has a duplicated label, if `duplicate_strategy` is
            "error" and a new column name is already taken, or if overwriting a name would drop
            the active geometry column.
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
        df = _safe_rename_columns(df, rename_columns, duplicate_strategy)

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
    A column whose title cased name is already taken is left under its original name.

    Args:
        df (AnyDataFrame): The input DataFrame.
        prefix (str): Column names prefixed with this value will be converted to title case.

    Returns:
        AnyDataFrame: The updated DataFrame.
    """

    mapping = {col: col.removeprefix(prefix).replace("_", " ").title() for col in df.columns if col.startswith(prefix)}
    df = _safe_rename_columns(df, mapping)

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
    A column whose stripped name is already taken is left under its original name.

    Args:
        df (AnyDataFrame): The input DataFrame.
        prefix (str): The prefix to remove from column names in this dataframe.

    Returns:
        AnyDataFrame: The updated DataFrame.
    """
    mapping = {col: col.removeprefix(prefix) for col in df.columns}
    df = _safe_rename_columns(df, mapping)
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
