import sqlite3
from typing import Annotated, Any, cast

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

# Functions a query may call. Anything else -- notably sqlite_version(),
# sqlite_source_id(), sqlite_compileoption_*() and load_extension() -- is refused
# by SQLite, so queries can't fingerprint the engine.
ALLOWED_FUNCTIONS = frozenset(
    {
        # aggregate
        "count", "sum", "total", "avg", "min", "max", "group_concat", "string_agg",
        # scalar
        "abs", "round", "coalesce", "ifnull", "nullif", "iif", "typeof",  # codespell:ignore iif
        "lower", "upper", "length", "substr", "substring", "trim", "ltrim", "rtrim",
        "replace", "instr", "like", "glob", "printf", "format", "concat", "concat_ws",
        # math
        "ceil", "ceiling", "floor", "sqrt", "pow", "power", "exp", "ln", "log", "log10",
        "mod", "sign",
        # date/time
        "date", "time", "datetime", "julianday", "strftime", "unixepoch",
        # json
        "json", "json_extract", "json_array_length", "json_type", "json_valid",
        "json_array", "json_object", "json_group_array", "json_group_object",
        # window
        "row_number", "rank", "dense_rank", "percent_rank", "cume_dist", "ntile",
        "lag", "lead", "first_value", "last_value", "nth_value",
    }
)  # fmt: skip

# The name the input DataFrame is registered under.
_TABLE_NAME = "df"

# Tables a query may read: the data, plus the JSON table-valued functions, which
# download workflows use to explode sanitized list columns into rows.
_READABLE_TABLES = frozenset({_TABLE_NAME, "json_each", "json_tree"})


def _authorizer(action: int, arg1: str | None, arg2: str | None, db_name: str | None, trigger: str | None) -> int:
    """Deny-by-default SQLite authorizer: plain/recursive SELECTs over `_READABLE_TABLES`, allowlisted functions."""
    if action in (sqlite3.SQLITE_SELECT, sqlite3.SQLITE_RECURSIVE):
        return sqlite3.SQLITE_OK
    if action == sqlite3.SQLITE_READ:
        # Also covers sqlite_stmt, sqlite_schema and the pragma_* eponymous virtual tables.
        return sqlite3.SQLITE_OK if arg1 in _READABLE_TABLES else sqlite3.SQLITE_DENY
    if action == sqlite3.SQLITE_FUNCTION:
        return sqlite3.SQLITE_OK if arg2 and arg2.lower() in ALLOWED_FUNCTIONS else sqlite3.SQLITE_DENY
    return sqlite3.SQLITE_DENY


def _run_sandboxed_query(query: str, df: AnyDataFrame) -> Any:  # pandasql is untyped
    """Run `query` against `df` via pandasql with `_authorizer` installed on the connection.

    `persist=True` hands us the connection before the query runs, so the table can
    be written (CREATE/INSERT) before the authorizer goes on; marking it loaded
    stops pandasql from rewriting it under the authorizer.
    """
    from pandasql.sqldf import PandaSQL, PandaSQLException, write_table  # type: ignore[import-untyped]

    pdsql = PandaSQL(persist=True)
    try:
        write_table(df, _TABLE_NAME, pdsql._conn)
        pdsql.loaded_tables.add(_TABLE_NAME)
        pdsql._conn.connection.dbapi_connection.set_authorizer(_authorizer)
        return pdsql(query, {_TABLE_NAME: df})
    except PandaSQLException as e:
        # `from None` keeps SQLAlchemy/pandasql internals out of the user-facing error.
        cause = e.args[0]
        raise ValueError(f"SQL query failed: {getattr(cause, 'orig', cause)}") from None
    finally:
        pdsql._conn.close()
        pdsql.engine.dispose()


def validate_sql_query(value: str) -> str:
    """
    Validate SQL query for security and safety.

    These are cheap early errors; what a query can actually touch is enforced by
    `_authorizer` at execution time. Syntax is not validated here.

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
        Queries run under a deny-by-default SQLite authorizer: only single SELECT
        statements (including CTEs) reading the 'df' table (and json_each/json_tree), calling functions in
        ALLOWED_FUNCTIONS. SQLite internals (sqlite_* functions, sqlite_schema,
        sqlite_stmt, pragma_* tables), PRAGMA, ATTACH, DDL and DML are refused.
        Queries are capped at 10,000 characters.

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
            original_crs = df.crs  # type: ignore[assignment]
            attrs = sanitize_for_arrow(df.drop(columns=[geom_name]))
            # Reattach geometry positionally rather than via `.join`.
            # which aligns on the index, and creates duplication in
            # cases where frames carry non-unique indexes
            attrs[geom_name] = df[geom_name].to_numpy()
            df = gpd.GeoDataFrame(attrs, geometry=geom_name, crs=original_crs)
        else:
            df = cast(AnyDataFrame, sanitize_for_arrow(df))

    if not query or not query.strip():
        return df

    # Filter columns if specified
    query_df = df[columns] if columns is not None else df

    if is_geodataframe and "geometry" in query_df.columns:
        # Store the original CRS
        original_crs = df.crs  # type: ignore[assignment]
        # Convert to regular DataFrame with WKT geometry for SQL processing
        query_df = query_df.copy()
        query_df["geometry"] = query_df["geometry"].apply(lambda geom: geom.wkt if geom else None)

    # Reset index to avoid duplicate column issues when index has a name
    query_df_reset = query_df.reset_index(drop=True)
    result = _run_sandboxed_query(query, query_df_reset)

    # If input was a GeoDataFrame and result has geometry column, convert back
    if is_geodataframe and "geometry" in result.columns:
        from shapely import wkt

        # Convert WKT back to geometry objects
        result["geometry"] = result["geometry"].apply(lambda x: wkt.loads(x) if x and isinstance(x, str) else None)
        # Create GeoDataFrame with original CRS
        result = gpd.GeoDataFrame(result, geometry="geometry", crs=original_crs)

    return cast(AnyDataFrame, result)
