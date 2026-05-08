import hashlib
import io
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field
from wt_registry import register

from ecoscope.platform.annotations import AnyDataFrame
from ecoscope.platform.serde import _persist_bytes, _persist_text


# TODO: Unlike the tasks in `._earthranger`, this is not tagged with `tags=["io"]`,
# because in the end to end test that tag is used to determine which tasks to mock.
# Ultimately, we should make the mocking process less brittle, but to get his PR merged,
# I'm going to leave this as is for now.
@register(
    deprecated=True,
    deprecation_message="Use persist_text_v2 instead, which accepts a configurable file extension.",
)
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
    """Persist text to a file or cloud storage object.

    Deprecated: hardcodes ``.html`` as the extension for auto-generated
    filenames. Use ``persist_text_v2`` to choose a different extension.
    """

    if not filename:
        # generate a filename if none is explicitly provided
        filename = hashlib.sha256(text.encode()).hexdigest()[:7] + ".html"
    if filename_suffix:
        filepath = Path(filename)
        filename = f"{filepath.stem}_{filename_suffix}{filepath.suffix}"

    return _persist_text(text, root_path, filename)


@register()
def persist_text_v2(
    text: Annotated[str, Field(description="Text to persist")],
    root_path: Annotated[str, Field(description="Root path to persist text to")],
    extension: Annotated[
        str,
        Field(
            description=(
                "File extension used when auto-generating a "
                "filename from a content hash. Ignored when `filename` is provided."
            ),
        ),
    ] = "html",
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
            If present, will be appended to the filename stem before the extension.
            """,
            exclude=True,
        ),
    ] = None,
) -> Annotated[str, Field(description="Path to persisted text")]:
    """Persist text to a file or cloud storage object."""

    if not filename:
        filename = hashlib.sha256(text.encode()).hexdigest()[:7] + f".{extension}"
    if filename_suffix:
        filepath = Path(filename)
        filename = f"{filepath.stem}_{filename_suffix}{filepath.suffix}"

    return _persist_text(text, root_path, filename)


FileType = Literal["csv", "gpkg", "geoparquet", "parquet", "geojson", "json"]


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
    import pandas as pd

    if not filename:
        # generate a filename if none is explicitly provided
        # Use repr of the dataframe shape and first few values to create a hash
        # This avoids issues with unhashable types in the dataframe
        try:
            hash_values = pd.util.hash_pandas_object(df).values
            # Convert to bytes - works for both ndarray and ExtensionArray
            hash_input = bytes(hash_values)
        except (TypeError, ValueError):
            # Fallback for unhashable types: use shape and first few rows
            content = f"{df.shape}{df.head(5).to_dict()}"
            hash_input = content.encode()
        filename = hashlib.sha256(hash_input).hexdigest()[:7]
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
