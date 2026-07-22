from collections import Counter
from typing import Annotated, Literal, cast

from pydantic import Field
from wt_registry import register

from ecoscope.platform.annotations import (  # type: ignore[import-untyped]
    AdvancedField,
    AnyDataFrame,
)


@register()
def drop_column_prefix(
    df: Annotated[
        AnyDataFrame,
        Field(
            description="The dataframe.",
            exclude=True,
        ),
    ],
    prefix: Annotated[str, Field(description="Drop the prefix.")],
    duplicate_strategy: Annotated[
        Literal["suffix", "error", "keep_original"],
        AdvancedField(
            description=(
                "Strategy for handling duplicate column names after removing prefix. "
                "'suffix': append _1, _2, etc. to duplicates; "
                "'error': raise ValueError if duplicates would occur; "
                "'keep_original': keep original name with prefix if duplicate would occur."
            ),
            default="suffix",
        ),
    ] = "suffix",
) -> AnyDataFrame:
    """
    Remove a prefix from column names in a DataFrame with duplicate handling.

    This task renames columns by removing a specified prefix from column names
    that start with that prefix. Other column names remain unchanged. Provides
    strategies for handling cases where removing the prefix would create duplicate
    column names.

    Args:
        df: Input DataFrame (modified in place)
        prefix: The prefix string to remove from column names
        duplicate_strategy: How to handle duplicate column names:
            - "suffix": Append _1, _2, etc. to duplicates (default)
            - "error": Raise ValueError if duplicates would occur
            - "keep_original": Keep original prefixed name if duplicate would occur

    Returns:
        The modified DataFrame with renamed columns

    Raises:
        ValueError: If duplicate_strategy is "error" and duplicates would be created

    Examples:
        >>> # Basic usage - no duplicates
        >>> df = pd.DataFrame({"prefix_col1": [1, 2], "prefix_col2": [3, 4], "other": [5, 6]})
        >>> result = drop_column_prefix(df, prefix="prefix_")
        >>> # Result columns: ["col1", "col2", "other"]

        >>> # Handling duplicates with suffix strategy (default)
        >>> df = pd.DataFrame({"prefix_name": [1, 2], "name": [3, 4], "prefix_value": [5, 6], "value": [7, 8]})
        >>> result = drop_column_prefix(df, prefix="prefix_", duplicate_strategy="suffix")
        >>> # Result columns: ["name_1", "name", "value_1", "value"]

        >>> # Handling duplicates with keep_original strategy
        >>> df = pd.DataFrame({"prefix_name": [1, 2], "name": [3, 4]})
        >>> result = drop_column_prefix(df, prefix="prefix_", duplicate_strategy="keep_original")
        >>> # Result columns: ["prefix_name", "name"]

        >>> # Handling duplicates with error strategy
        >>> df = pd.DataFrame({"prefix_name": [1, 2], "name": [3, 4]})
        >>> result = drop_column_prefix(df, prefix="prefix_", duplicate_strategy="error")
        >>> # Raises: ValueError("Removing prefix would create duplicate columns: ['name']")
    """
    # Build mapping of columns to rename
    columns_to_rename = {col: col[len(prefix) :] for col in df.columns if col.startswith(prefix)}

    if not columns_to_rename:
        return cast(AnyDataFrame, df)

    # Detect potential duplicates
    new_names = list(columns_to_rename.values())
    existing_names = set(df.columns) - set(columns_to_rename.keys())
    all_final_names = list(existing_names) + new_names

    # Find duplicates (names that appear more than once in final column list)
    # Use case-insensitive comparison because formats like GPKG/SQLite treat
    # column names case-insensitively (e.g. "Vessel ID" vs "Vessel Id" collide)
    seen: dict[str, int] = {}
    duplicates = set()
    for name in all_final_names:
        key = name.lower()
        if key in seen:
            duplicates.add(key)
        seen[key] = seen.get(key, 0) + 1

    if duplicates:
        if duplicate_strategy == "error":
            raise ValueError(f"Removing prefix would create duplicate columns: {sorted(duplicates)}")
        elif duplicate_strategy == "keep_original":
            # Don't rename columns that would create duplicates
            final_mapping = {}
            used_names = {col.lower() for col in existing_names}
            for old_col, new_col in columns_to_rename.items():
                if new_col.lower() not in used_names:
                    final_mapping[old_col] = new_col
                    used_names.add(new_col.lower())
            df.rename(columns=final_mapping, inplace=True)
        elif duplicate_strategy == "suffix":
            # Add numeric suffixes to handle duplicates (case-insensitive)
            final_mapping = {}
            name_counts = Counter(col.lower() for col in existing_names)

            # Second pass: rename columns with suffixes when needed
            for old_col, new_col in columns_to_rename.items():
                key = new_col.lower()
                if key in name_counts:
                    # This name already exists, add suffix
                    count = name_counts[key]
                    final_mapping[old_col] = f"{new_col}_{count}"
                    name_counts[key] = count + 1
                else:
                    # Name is unique, use as-is
                    final_mapping[old_col] = new_col
                    name_counts[key] = 1

            df.rename(columns=final_mapping, inplace=True)
    else:
        # No duplicates, proceed with simple rename
        df.rename(columns=columns_to_rename, inplace=True)

    return cast(AnyDataFrame, df)


@register()
def drop_duplicate_columns(
    df: Annotated[
        AnyDataFrame,
        Field(
            description="The dataframe.",
            exclude=True,
        ),
    ],
    strategy: Annotated[
        Literal["drop_first", "drop_last", "suffix"],
        Field(
            description=(
                "Strategy for handling duplicate column names. "
                "'drop_first': keep the last occurrence; "
                "'drop_last': keep the first occurrence; "
                "'suffix': rename duplicates with _1, _2, etc."
            ),
            default="drop_last",
        ),
    ] = "drop_last",
) -> AnyDataFrame:
    """
    Detect and resolve duplicate column names in a DataFrame.

    Args:
        df: Input DataFrame that may contain duplicate column names.
        strategy: How to handle duplicates:
            - "drop_first": Keep the last occurrence, drop earlier ones
            - "drop_last": Keep the first occurrence, drop later ones
            - "suffix": Rename duplicates with _1, _2, etc. suffixes

    Returns:
        DataFrame with unique column names.

    Examples:
        >>> df = pd.DataFrame([[1, 2, 3]], columns=["A", "B", "A"])
        >>> result = drop_duplicate_columns(df, strategy="drop_last")
        >>> # Result columns: ["A", "B"] — keeps first "A"

        >>> result = drop_duplicate_columns(df, strategy="drop_first")
        >>> # Result columns: ["B", "A"] — keeps last "A"

        >>> result = drop_duplicate_columns(df, strategy="suffix")
        >>> # Result columns: ["A", "B", "A_1"] — renames duplicate
    """
    columns = list(df.columns)
    if len(columns) == len(set(columns)):
        return cast(AnyDataFrame, df)

    if strategy == "drop_first":
        # Keep last occurrence: reverse, drop_duplicates keeping first, reverse back
        df = df.loc[:, ~df.columns[::-1].duplicated()[::-1]]
    elif strategy == "drop_last":
        # Keep first occurrence
        df = df.loc[:, ~df.columns.duplicated()]
    elif strategy == "suffix":
        counts: Counter[str] = Counter()
        new_columns = []
        for col in columns:
            if counts[col] > 0:
                new_columns.append(f"{col}_{counts[col]}")
            else:
                new_columns.append(col)
            counts[col] += 1
        df.columns = new_columns

    return cast(AnyDataFrame, df)
