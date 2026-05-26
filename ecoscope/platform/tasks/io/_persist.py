import hashlib
import io
import json
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field
from wt_registry import register

from ecoscope.platform.annotations import AnyDataFrame
from ecoscope.platform.serde import _persist_bytes, _persist_text


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


def _iso_format_timestamp_columns(df):
    """Convert pandas datetime64[*] columns (tz-naive and tz-aware) to ISO-
    8601 string columns. The resulting parquet has Utf8 columns where the
    gdf had Timestamp columns — tooltips and other JS-side display
    consumers see plain strings and don't need to know about Arrow
    Timestamp units or BigInt nanosecond arithmetic at display time.

    Loses sub-second precision (acceptable for display). If sub-second
    precision matters in the source gdf for some future consumer, widen
    the format string (e.g. `%S.%f`) or skip this transformation for that
    case.
    """
    import pandas as pd

    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col].dtype):
            df[col] = df[col].dt.strftime("%Y-%m-%dT%H:%M:%S%z")


def _downcast_float_columns(df):
    """Re-encode float64 columns as float32. @geoarrow/deck.gl-geoarrow hands
    scalar accessor columns (radius, width, weight, elevation) through to
    deck.gl as binary attributes, and the underlying layers' attribute system
    rejects Float64Array — Float32Array is the expected typed array for these
    accessors. Geometry coordinate columns aren't affected (GeoArrow encodes
    them as fixed-size lists / structs, not plain float columns).
    """
    import pandas as pd

    for col in df.columns:
        if pd.api.types.is_float_dtype(df[col].dtype) and df[col].dtype == "float64":
            df[col] = df[col].astype("float32")


def _pack_color_columns(df):
    """Re-encode object-dtype columns of length-3 or length-4 int tuples in
    the 0-255 range as pyarrow FixedSizeList<Uint8>[3|4] — the shape
    @geoarrow/deck.gl-geoarrow color accessors require. Anything that
    doesn't match that exact RGB(A) shape (wrong length, non-int, out of
    range, ragged) passes through with pyarrow's default List inference.

    Raises if a color-shaped column has null rows: parquet's nested-type
    encoding can't represent a FixedSizeList with null elements (pyarrow
    rejects with "Lists with non-zero length null components are not
    supported"), and there's no writer option that works around it.
    Callers should fill null rows with a sentinel (e.g. transparent black
    `(0, 0, 0, 0)`) before persisting.
    """
    import numpy as np
    import pandas as pd
    import pyarrow as pa

    for col in df.columns:
        if df[col].dtype != "object":
            continue
        non_null = df[col].dropna()
        if non_null.empty or not isinstance(non_null.iloc[0], (tuple, list)):
            continue
        try:
            sample = np.asarray(non_null.tolist())
        except ValueError:
            continue  # ragged / mixed lengths
        if (
            sample.ndim != 2
            or sample.dtype.kind not in "iu"
            or sample.shape[1] not in (3, 4)
            or sample.min() < 0
            or sample.max() > 255
        ):
            continue
        if len(non_null) != len(df):
            raise ValueError(
                f"Color column {col!r} contains null values, which parquet's "
                f"FixedSizeList encoding can't represent. Fill null rows with "
                f"a sentinel color (e.g. (0, 0, 0, 0) for transparent) before "
                f"persisting."
            )
        fsl = pa.array(df[col].tolist(), type=pa.list_(pa.uint8(), sample.shape[1]))
        df[col] = pd.arrays.ArrowExtensionArray(fsl)


@register()
def persist_arrow(
    df: Annotated[AnyDataFrame, Field(description="Dataframe to persist as GeoArrow-encoded parquet")],
    root_path: Annotated[str, Field(description="Root path to persist parquet to")],
    filename: Annotated[
        str | None,
        Field(
            description=(
                "Optional filename within `root_path`. Auto-generated from a "
                "df content hash if absent. The `.parquet` extension is "
                "appended automatically."
            ),
            exclude=True,
        ),
    ] = None,
) -> Annotated[str, Field(description="Path to persisted parquet")]:
    """Persist a dataframe as GeoArrow-encoded parquet, ready for
    @geoarrow/deck.gl-geoarrow layers.

    Differs from `persist_df(filetype='geoparquet')` in three ways:

    - Geometry is encoded via GeoArrow extension types (`geoarrow.point`,
      `geoarrow.polygon`, etc.) rather than the default WKB blobs.
      Required for the JS-side GeoArrow layers to auto-detect the
      geometry column.
    - Object-dtype columns of length-3 or length-4 int tuples in [0, 255]
      are re-encoded as pyarrow FixedSizeList<Uint8>[3|4] — the shape the
      GeoArrow color accessor (`getFillColor` / `getLineColor`) requires.
      pandas has no native fixed-size-list dtype, so a plain RGB(A) tuple
      column otherwise round-trips through parquet as `List<Int64>`, which
      the layer's `validateColorVector` rejects.
    - Datetime columns (`datetime64[ns]` / `datetime64[ns, tz]`) are
      converted to ISO-8601 string columns. The JS layers and tooltip see
      plain strings; the parquet's Timestamp type info is not preserved.
      Acceptable because `persist_arrow` targets the GeoArrow rendering
      pipeline specifically.

    Use `persist_df(filetype='geoparquet')` instead when writing for
    WKB-expecting consumers (e.g. QGIS, PostGIS) or when you don't want
    the color / timestamp repacking.
    """
    import geopandas as gpd  # type: ignore[import-untyped]
    import pandas as pd

    if not filename:
        try:
            hash_input = bytes(pd.util.hash_pandas_object(df).values)
        except (TypeError, ValueError):
            hash_input = f"{df.shape}{df.head(5).to_dict()}".encode()
        filename = hashlib.sha256(hash_input).hexdigest()[:7]

    gdf = gpd.GeoDataFrame(df).copy()
    _iso_format_timestamp_columns(gdf)
    _downcast_float_columns(gdf)
    _pack_color_columns(gdf)
    buffer = io.BytesIO()
    gdf.to_parquet(buffer, index=False, geometry_encoding="geoarrow")
    return _persist_bytes(buffer.getvalue(), root_path, f"{filename}.parquet")
