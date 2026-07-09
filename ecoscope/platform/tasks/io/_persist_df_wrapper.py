import hashlib
import json
from typing import Annotated, Literal, cast

import geopandas as gpd  # type: ignore[import-untyped]
import numpy as np
import pandas as pd
from pydantic import Field
from pydantic.json_schema import SkipJsonSchema
from wt_registry import register
from wt_task.skip import SkippedDependencyFallback, SkipSentinel

from ecoscope.platform.annotations import (  # type: ignore[import-untyped]
    AdvancedField,
    AnyDataFrame,
)
from ecoscope.platform.indexes import (  # type: ignore[import-untyped]
    CompositeFilter,
)
from ecoscope.platform.tasks.transformation._sanitize import (
    sanitize_for_arrow,
)

FileType = Literal["csv", "gpkg", "geoparquet", "parquet"]


def _fallback_to_empty(df: AnyDataFrame | SkipSentinel) -> AnyDataFrame:
    """Fallback function to convert SkipSentinel to an empty df."""
    return cast(AnyDataFrame, pd.DataFrame({"empty_event": []})) if isinstance(df, SkipSentinel) else df


@register()
def persist_df_wrapper(
    df: Annotated[
        AnyDataFrame,
        Field(description="Dataframe to persist"),
        SkippedDependencyFallback(_fallback_to_empty),
    ],
    root_path: Annotated[str, Field(description="Root path to persist text to")],
    filename: Annotated[
        str | SkipJsonSchema[None],
        Field(
            description="""\
            Optional filename to persist text to within the `root_path`.
            If not provided, a filename will be generated based on a hash of the df content.
            """,
            default=None,
            exclude=True,
        ),
    ] = None,
    filetypes: Annotated[
        list[FileType] | SkipJsonSchema[None],
        Field(
            description="The output format",
            default=["csv"],
            json_schema_extra={"uniqueItems": True},
        ),
    ] = None,
    filename_prefix: Annotated[
        str | SkipJsonSchema[None],
        AdvancedField(
            description="""\
            Optional filename prefix to persist text to within the `root_path`.
            We will always add a suffix based on the dataframe content hash to avoid duplicates.
            """,
            default=None,
        ),
    ] = None,
    sanitize: Annotated[
        bool,
        AdvancedField(
            description="""\
            Whether to sanitize the dataframe for Arrow compatibility before persisting,
            recommended when including event or observation details
            """,
            default=False,
        ),
    ] = False,
) -> Annotated[list[str], Field(description="A list of paths to persisted data")]:
    """
    Save a DataFrame to disk with optional sanitization for Arrow compatibility.

    This task wraps the core persist_df function and adds optional data sanitization
    to ensure compatibility with Arrow-based formats (Parquet, GeoParquet). When
    sanitize=True, the task converts complex Python objects (lists, dicts, sets) to
    JSON strings, normalizes data types, and handles mixed-type columns.

    The sanitization process:
    - Converts bytes to UTF-8 strings
    - Serializes collections (list, dict, set) to JSON strings
    - Infers numeric types and uses pandas nullable Int64 where appropriate
    - Converts mixed-type columns to strings
    - Handles GeoDataFrames by preserving geometry columns

    Args:
        df: DataFrame or GeoDataFrame to save
        root_path: Directory where the file will be saved
        filename: Optional filename (without path). If None, generates hash-based name.
        filetypes: A list of output formats - "csv", "gpkg", or "geoparquet"
        sanitize: Whether to sanitize data for Arrow compatibility (default: False)

    Returns:
        Full path to the saved file

    Example:
        >>> # Save with complex data types
        >>> df = pd.DataFrame({
        ...     "name": ["A", "B"],
        ...     "tags": [["tag1", "tag2"], ["tag3"]],
        ...     "metadata": [{"key": "value"}, {"x": 1}]
        ... })
        >>> path = persist_df_wrapper(
        ...     df=df,
        ...     root_path="/output",
        ...     filename="data.csv",
        ...     filetypes=["csv"],
        ...     sanitize=True  # Converts lists/dicts to JSON strings
        ... )
    """
    import geopandas as gpd  # type: ignore[import-untyped]

    from ecoscope.platform.tasks.io._persist import (  # type: ignore[import-untyped]
        persist_df,
    )

    if not filetypes:
        filetypes = ["csv"]
    if sanitize:
        if isinstance(df, gpd.GeoDataFrame):
            geom_name = df.geometry.name
            attrs = df.drop(columns=[geom_name])
            attrs = cast(gpd.GeoDataFrame, sanitize_for_arrow(attrs))
            # let geopandas handle geometry encoding
            df_new = cast(AnyDataFrame, attrs.join(df[[geom_name]]))
        else:
            df_new = cast(AnyDataFrame, sanitize_for_arrow(df))
    else:
        df_new = df

    # generate a filehash
    # Use repr of the dataframe shape and first few values to create a hash
    # This avoids issues with unhashable types in the dataframe
    try:
        hash_values = pd.util.hash_pandas_object(df).values
        # Convert to bytes - handle both ndarray and ExtensionArray
        if isinstance(hash_values, np.ndarray):
            hash_input = hash_values.tobytes()
        else:
            # ExtensionArray - convert to numpy array first
            hash_input = np.asarray(hash_values).tobytes()
    except (TypeError, ValueError, AttributeError):
        # Fallback for unhashable types: use shape and first few rows
        content = f"{df.shape}{df.head(5).to_dict()}"
        hash_input = content.encode()

    filehash = hashlib.sha256(hash_input).hexdigest()[:7]
    filename = f"{filename_prefix}_{filehash}" if filename_prefix else filehash

    paths = [persist_df(df_new, root_path, filename, filetype) for filetype in filetypes]

    return paths


def _hash_grouper_key(composite_filter: CompositeFilter) -> str:
    """Return a 7-char sha256 hash of the JSON-encoded grouper key.

    The CompositeFilter is first reduced to a `{column: value}` dict, e.g.
    `(("patrol_type", "=", "Routine Patrol"),)` -> `{"patrol_type": "Routine Patrol"}`,
    matching the dashboard's `views_json` key encoding so the FE can recompute
    the same hash and match files to views.
    """
    json_key = {cond[0]: cond[2] for cond in composite_filter}
    return hashlib.sha256(json.dumps(json_key, sort_keys=True).encode()).hexdigest()[:6]


def _fallback_to_empty_grouped(
    iterable: "list[tuple[CompositeFilter, AnyDataFrame]] | SkipSentinel",
) -> "list[tuple[CompositeFilter, AnyDataFrame]]":
    """Fallback for skipped upstream dependency: produce an empty keyed iterable."""
    if isinstance(iterable, SkipSentinel):
        return []
    return [(key, _fallback_to_empty(df)) for key, df in iterable]


@register()
def persist_grouped_dfs_for_results_download(
    grouped_dfs: Annotated[
        list[tuple[CompositeFilter | None, AnyDataFrame]],
        Field(
            description=(
                "A keyed iterable of (group key, dataframe) tuples as produced by `split_groups`. "
                "Each dataframe is persisted to its own file(s) with a 7-char "
                "hash of the associated group key embedded in the filename."
            ),
        ),
        SkippedDependencyFallback(_fallback_to_empty_grouped),
    ],
    root_path: Annotated[str, Field(description="Root path to persist dataframes to")],
    filetypes: Annotated[
        list[FileType] | SkipJsonSchema[None],
        Field(
            description="The output format",
            default=["csv"],
            json_schema_extra={"uniqueItems": True},
        ),
    ] = None,
    filename_prefix: Annotated[
        str | SkipJsonSchema[None],
        AdvancedField(
            description="""\
            Optional filename to persist text to within the `root_path`.
            If not provided, a filename will be generated based on a hash of the df content.
            """,
            default=None,
        ),
    ] = None,
    sanitize: Annotated[
        bool,
        AdvancedField(
            description="""\
            Whether to sanitize each dataframe for Arrow compatibility before
            persisting, recommended when including event or observation details
            """,
            default=False,
        ),
    ] = False,
) -> Annotated[
    list[str],
    Field(description="A flat list of paths to all persisted group files"),
]:
    """Persist grouped or ungrouped dataframes for FE results-download.

    Delegates to `persist_df_wrapper` per dataframe, prefixing the filename
    with a 7-char sha256 hash of the encoded group key so the FE can match
    files to dashboard views. Final filename layout:

        [<filename_prefix>_]<key_hash>_<df_hash>.<extension>

    For a keyed iterable input, each group's own CompositeFilter is hashed.
    For a single dataframe input, the file is hashed under the default view
    key the FE will pre-select — derived from `groupers` + `reference_keys`
    via the same `composite_filters_to_grouper_choices_dict` the dashboard
    uses. Both must be provided in the ungrouped case.
    """
    paths: list[str] = []
    for composite_filter, df in grouped_dfs:
        # grouped_df's SkippedDependencyFallback gives empty dfs
        if composite_filter is not None and not df.empty:
            key_hash = _hash_grouper_key(composite_filter)
            prefix_and_group_hash = f"{filename_prefix}_{key_hash}" if filename_prefix else key_hash
            paths.extend(
                persist_df_wrapper(
                    df=df,
                    root_path=root_path,
                    filetypes=filetypes,
                    filename_prefix=prefix_and_group_hash,
                    sanitize=sanitize,
                )
            )

    return paths
