import hashlib
import io
import json
from pathlib import Path
from typing import Annotated, Any, Literal, cast

from pydantic import BaseModel, Field
from pydantic.json_schema import SkipJsonSchema
from wt_registry import register
from wt_task.skip import SkippedDependencyFallback, SkipSentinel

from ecoscope.platform.annotations import AdvancedField, AnyDataFrame
from ecoscope.platform.indexes import CompositeFilter
from ecoscope.platform.serde import _persist_bytes, _persist_text
from ecoscope.platform.tasks.transformation._sanitize import sanitize_for_arrow


# TODO: Unlike the tasks in `._earthranger`, this is not tagged with `tags=["io"]`,
# because in the end to end test that tag is used to determine which tasks to mock.
# Ultimately, we should make the mocking process less brittle, but to get his PR merged,
# I'm going to leave this as is for now.
@register()
def persist_text(
    text: Annotated[str, Field(description="Text to persist")],
    root_path: Annotated[str, Field(description="Root path to persist text to")],
    filename: Annotated[
        str | None,
        Field(
            description="""\
            Optional filename to persist text to within the `root_path`.
            If not provided, a filename will be generated based on a hash of the text content.
            """,
            exclude=True,
        ),
    ] = None,
    filename_suffix: Annotated[
        str | None,
        Field(
            description="""\
            If present, will be appended to the filename as filename_suffix.html
            """,
            exclude=True,
        ),
    ] = None,
) -> Annotated[str, Field(description="Path to persisted text")]:
    """Persist text to a file or cloud storage object."""

    if not filename:
        # generate a filename if none is explicitly provided
        filename = hashlib.sha256(text.encode()).hexdigest()[:7] + ".html"
    if filename_suffix:
        filepath = Path(filename)
        filename = f"{filepath.stem}_{filename_suffix}{filepath.suffix}"

    return _persist_text(text, root_path, filename)


@register()
def persist_json(
    data: Annotated[
        dict[str, Any] | BaseModel,
        Field(description="JSON-serializable dict or pydantic model to persist"),
    ],
    root_path: Annotated[str, Field(description="Root path to persist data to")],
    filename: Annotated[
        str | None,
        Field(
            description="""\
            Optional filename (within `root_path`). If not provided, a filename will
            be generated from a hash of the serialized JSON content. The `.json`
            extension is appended automatically when one isn't already present.
            """,
            exclude=True,
        ),
    ] = None,
    filename_suffix: Annotated[
        str | None,
        Field(
            description="If present, appended to the filename stem before the extension.",
            exclude=True,
        ),
    ] = None,
) -> Annotated[str, Field(description="Path to persisted JSON")]:
    """Serialize JSON-shaped data and persist to a file or cloud storage object."""
    if isinstance(data, BaseModel):
        payload = data.model_dump_json()
    else:
        payload = json.dumps(data)

    if not filename:
        filename = hashlib.sha256(payload.encode()).hexdigest()[:7] + ".json"
    elif not Path(filename).suffix:
        filename = f"{filename}.json"
    if filename_suffix:
        filepath = Path(filename)
        filename = f"{filepath.stem}_{filename_suffix}{filepath.suffix}"

    return _persist_text(payload, root_path, filename)


FileType = Literal["csv", "gpkg", "geoparquet", "parquet", "geojson", "json"]

# Formats supported by the dataframe-persistence wrapper tasks below. A subset of
# `FileType` — the round-trippable tabular/geo formats appropriate for a full
# dataframe download (raw `geojson`/`json` text dumps are intentionally excluded).
ResultsFileType = Literal["csv", "gpkg", "geoparquet", "parquet"]


def _hash_df(df: AnyDataFrame) -> str:
    """Return a 7-char sha256 hash of a dataframe's contents.

    Falls back to hashing the shape + first rows when the frame holds unhashable
    values. Shared by `persist_df` and `persist_df_wrapper` so both derive
    identical content-based filenames.
    """
    import numpy as np
    import pandas as pd

    # Use a hash of the dataframe content; avoids issues with unhashable types.
    try:
        hash_values = pd.util.hash_pandas_object(df).values
        # Convert to bytes - handle both ndarray and ExtensionArray
        if isinstance(hash_values, np.ndarray):
            hash_input = hash_values.tobytes()
        else:
            hash_input = np.asarray(hash_values).tobytes()
    except (TypeError, ValueError, AttributeError):
        # Fallback for unhashable types: use shape and first few rows
        content = f"{df.shape}{df.head(5).to_dict()}"
        hash_input = content.encode()
    return hashlib.sha256(hash_input).hexdigest()[:7]


@register()
def persist_df(
    df: Annotated[AnyDataFrame, Field(description="Dataframe to persist")],
    root_path: Annotated[str, Field(description="Root path to persist text to")],
    filename: Annotated[
        str | None,
        Field(
            description="""\
            Optional filename to persist text to within the `root_path`.
            If not provided, a filename will be generated based on a hash of the df content.
            """,
        ),
    ] = None,
    filetype: Annotated[FileType, Field(description="The output format")] = "csv",
) -> Annotated[str, Field(description="Path to persisted data")]:
    """Persist dataframe to a file or cloud storage object."""
    import geopandas as gpd  # type: ignore[import-untyped]

    if not filename:
        # generate a filename if none is explicitly provided
        filename = _hash_df(df)
    if filetype == "csv":
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer)
        return _persist_text(csv_buffer.getvalue(), root_path, f"{filename}.{filetype}")
    elif filetype == "gpkg":
        buffer = io.BytesIO()
        gdf = gpd.GeoDataFrame(df)
        gdf.to_file(buffer, driver="GPKG")
        return _persist_bytes(buffer.getvalue(), root_path, f"{filename}.{filetype}")
    elif filetype == "geoparquet":
        buffer = io.BytesIO()
        gdf = gpd.GeoDataFrame(df)
        gdf.to_parquet(buffer, index=False)
        return _persist_bytes(buffer.getvalue(), root_path, f"{filename}.parquet")
    elif filetype == "parquet":
        buffer = io.BytesIO()
        has_geom = any(isinstance(df[col].dtype, gpd.array.GeometryDtype) for col in df.columns)
        if has_geom:
            gpd.GeoDataFrame(df).to_parquet(buffer, index=False)
        else:
            df.to_parquet(buffer, index=False)
        return _persist_bytes(buffer.getvalue(), root_path, f"{filename}.parquet")
    elif filetype == "geojson":
        gdf = gpd.GeoDataFrame(df)
        return _persist_text(gdf.to_json(), root_path, f"{filename}.{filetype}")
    elif filetype == "json":
        return _persist_text(df.to_json(), root_path, f"{filename}.{filetype}")
    else:
        raise ValueError(f"Unsupported file type: {filetype}")


def _fallback_to_empty(df: AnyDataFrame | SkipSentinel) -> AnyDataFrame:
    """Fallback function to convert SkipSentinel to an empty df."""
    import pandas as pd

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
        list[ResultsFileType] | SkipJsonSchema[None],
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

    Builds on `persist_df` by adding optional data sanitization (for Arrow-based
    formats like Parquet/GeoParquet), multi-filetype output, and a
    `SkippedDependencyFallback`. When sanitize=True, complex Python objects
    (lists, dicts, sets) are converted to JSON strings, data types are
    normalized, and mixed-type columns are handled.

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
        filetypes: A list of output formats - "csv", "gpkg", "geoparquet", or "parquet"
        sanitize: Whether to sanitize data for Arrow compatibility (default: False)

    Returns:
        A list of paths to the saved files (one per filetype).

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

    filehash = _hash_df(df)
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
        list[ResultsFileType] | SkipJsonSchema[None],
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

    Each group's own CompositeFilter is hashed. Groups with a `None` key (e.g.
    the SkippedDependencyFallback sentinel) or an empty dataframe are skipped.
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
