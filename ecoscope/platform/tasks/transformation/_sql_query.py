from typing import Annotated, cast

from pydantic import AfterValidator, Field
from pydantic.json_schema import SkipJsonSchema
from wt_registry import register

from ecoscope.platform.annotations import (  # type: ignore[import-untyped]
    AdvancedField,
    AnyDataFrame,
)

from ._sanitize import sanitize_for_arrow

# Maximum allowed query length (characters)
MAX_QUERY_LENGTH = 10_000

# Dangerous SQL keywords that are blocked
BLOCKED_KEYWORDS = [
    "DROP",
    "CREATE",
    "ALTER",
    "TRUNCATE",
    "INSERT",
    "UPDATE",
    "DELETE",
    "REPLACE",
    "PRAGMA",
    "ATTACH",
]


def validate_sql_query(value: str) -> str:
    """
    Validate SQL query for security and safety.

    This function checks that the query is safe to execute by:
    - Ensuring it's not empty
    - Blocking dangerous SQL operations (DDL/DML)
    - Blocking SQL comments and multiple statements
    - Requiring SELECT statements only
    - Enforcing maximum query length

    We do NOT validate SQL syntax here to reduce the validation overhead

    Args:
        value: SQL query string to validate

    Returns:
        The validated query string (stripped of leading/trailing whitespace)

    Raises:
        ValueError: If the query fails validation
    """

    if not value or not value.strip():
        return value.strip()

    # Check query length
    if len(value) > MAX_QUERY_LENGTH:
        raise ValueError(f"Query exceeds maximum length of {MAX_QUERY_LENGTH} characters")

    # Strip and prepare for checking
    query_stripped = value.strip()
    query_upper = query_stripped.upper()

    # Block multiple statements (semicolon not at end)
    # Allow semicolon at the very end, but not in the middle
    semicolon_pos = query_stripped.find(";")
    if semicolon_pos != -1 and semicolon_pos < len(query_stripped) - 1:
        # There's a semicolon that's not at the end
        raise ValueError("Multiple SQL statements are not allowed (no semicolons)")

    # Check for blocked keywords
    for keyword in BLOCKED_KEYWORDS:
        if keyword in query_upper:
            raise ValueError(f"Forbidden SQL keyword detected: {keyword}")

    # Must start with SELECT
    if not query_upper.startswith("SELECT"):
        raise ValueError("Only SELECT queries are allowed")

    return query_stripped


@register()
def apply_sql_query(
    df: Annotated[
        AnyDataFrame,
        Field(
            description="The input DataFrame to query.",
            exclude=True,
        ),
    ],
    query: Annotated[
        str,
        AdvancedField(
            description="SQL query string to apply to the DataFrame. Leaves it unchanged when the field is empty"
            "Use 'df' as the table name in the query.",
            title="SQL Query",
            default="",
        ),
        AfterValidator(validate_sql_query),
    ] = "",
    columns: Annotated[
        list[str] | SkipJsonSchema[None],
        AdvancedField(
            description="Optional list of column names to include in the SQL query context. "
            "If specified, only these columns will be available in the 'df' table for querying. "
            "Use this to exclude columns with unsupported data types (list, dict) that cannot be "
            "stored in SQLite. If not specified, all columns are included.",
            default=None,
        ),
    ] = None,
    sanitize: Annotated[
        bool,
        AdvancedField(
            description="Whether to sanitize the DataFrame for Arrow/SQLite compatibility "
            "before querying. When True (default), complex columns (list, dict, set, bytes) "
            "are converted to JSON strings so pandasql/SQLite accepts them, removing the need "
            "for the 'columns' whitelist. Geometry columns are preserved. Set to False to pass "
            "columns through untouched.",
            default=True,
        ),
    ] = True,
) -> AnyDataFrame:
    """
    Apply a SQL query to a DataFrame using pandasql.

    This task allows you to filter, aggregate, transform, and manipulate DataFrames
    using familiar SQL syntax. The DataFrame is registered as a table named 'df'
    in the query context.

    The task uses pandasql (SQLite) as the SQL engine, which provides:
    - Full SQL support (SELECT, WHERE, JOIN, GROUP BY, ORDER BY, HAVING, etc.)
    - Familiar SQL syntax for data analysts
    - Support for both pandas DataFrames and GeoPandas GeoDataFrames
    - Subqueries, UNION, and complex expressions

    Data Type Limitations:
        SQLite does not support complex Python types (list, dict, set, bytes, nested
        structures). By default (sanitize=True) these columns are automatically converted
        to JSON strings before the query runs, so SQL "just works" without preprocessing.

        Set sanitize=False to pass columns through untouched (e.g. when a downstream task
        relies on real list/dict values). In that case, use the 'columns' parameter to
        exclude unsupported columns, or normalize them (e.g. using the normalize_json_column
        task) before applying SQL queries.

    Security Notes:
        This task validates queries to prevent dangerous operations:
        - Only SELECT statements are allowed
        - DDL/DML operations (DROP, INSERT, UPDATE, DELETE) are blocked
        - SQL comments (--) and multiple statements (;) are not permitted
        - Queries have a maximum length limit of 10,000 characters

        However, this task assumes trusted users are authoring workflows.
        Do not use with untrusted or user-provided SQL strings.

    Performance Notes:
        - For very large DataFrames (>1M rows), native pandas operations may be faster
        - pandasql creates temporary SQLite tables, which adds overhead
        - Best suited for complex queries that would be difficult to express in pandas

    Args:
        df: Input DataFrame to query (registered as table 'df' in the query)
        query: SQL query string (e.g., "SELECT * FROM df WHERE column > 10")
        columns: Optional list of column names to include. Use this to exclude columns
                 with unsupported types (list, dict). If None, all columns are included.
        sanitize: When True (default), convert complex columns (list, dict, set, bytes)
                 to JSON strings before querying so SQLite accepts them; geometry columns
                 are preserved. Set to False to pass columns through untouched.

    Returns:
        DataFrame resulting from the SQL query execution. If the input was a
        GeoDataFrame and the geometry column is included in the SELECT, the
        result will also be a GeoDataFrame with the original CRS preserved.
    """
    import geopandas as gpd  # type: ignore[import-untyped]
    from pyproj.crs import CRS

    is_geodataframe = isinstance(df, gpd.GeoDataFrame)
    original_crs: CRS | None = None

    # Sanitize complex columns (list, dict, set, bytes) to JSON strings so
    # pandasql/SQLite accepts them. Done before the empty-query early-return so
    # the returned DataFrame is consistent regardless of whether a query runs.
    if sanitize:
        if is_geodataframe:
            geom_name = df.geometry.name
            original_crs = df.crs  # type: ignore[assignment] - mypy doesn't see this is a CRS
            attrs = sanitize_for_arrow(df.drop(columns=[geom_name]))
            # Preserve the geometry column; the WKT round-trip below handles it.
            df = gpd.GeoDataFrame(attrs.join(df[[geom_name]]), geometry=geom_name, crs=original_crs)
        else:
            df = cast(AnyDataFrame, sanitize_for_arrow(df))

    if not query or not query.strip():
        return df

    from pandasql import sqldf  # type: ignore[import-untyped]

    # Filter columns if specified
    query_df = df[columns] if columns is not None else df

    if is_geodataframe and "geometry" in query_df.columns:
        # Store the original CRS
        original_crs = df.crs  # type: ignore[assignment] - mypy doesn't see this is a CRS
        # Convert to regular DataFrame with WKT geometry for SQL processing
        query_df = query_df.copy()
        query_df["geometry"] = query_df["geometry"].apply(lambda geom: geom.wkt if geom else None)

    # Execute the query
    # sqldf expects a dictionary of DataFrames, with 'df' as the key
    # Reset index to avoid duplicate column issues when index has a name
    query_df_reset = query_df.reset_index(drop=True)
    result = sqldf(query, {"df": query_df_reset})

    # If input was a GeoDataFrame and result has geometry column, convert back
    if is_geodataframe and "geometry" in result.columns:
        from shapely import wkt

        # Convert WKT back to geometry objects
        result["geometry"] = result["geometry"].apply(lambda x: wkt.loads(x) if x and isinstance(x, str) else None)
        # Create GeoDataFrame with original CRS
        result = gpd.GeoDataFrame(result, geometry="geometry", crs=original_crs)

    return cast(AnyDataFrame, result)
