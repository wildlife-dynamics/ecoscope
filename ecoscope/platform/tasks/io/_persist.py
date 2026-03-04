import hashlib
import io
from pathlib import Path
from typing import Annotated, Literal

from ecoscope.platform.annotations import AnyDataFrame
from ecoscope.platform.serde import _persist_bytes, _persist_text
from pydantic import Field
from wt_registry import register


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


FileType = Literal["csv", "gpkg", "geoparquet"]


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
    else:
        raise ValueError(f"Unsupported file type: {filetype}")
