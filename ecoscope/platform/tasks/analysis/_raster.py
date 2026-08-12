import hashlib
import json
import logging
import os
from typing import Annotated, Optional, TypeAlias
from urllib.parse import urlparse

import numpy as np  # type: ignore[import-untyped]
from pydantic import Field
from wt_registry import register

from ecoscope.io.raster import RasterData  # type: ignore[import-untyped]
from ecoscope.platform.annotations import AdvancedField  # type: ignore[import-untyped]
from ecoscope.platform.indexes import CompositeFilter  # type: ignore[import-untyped]
from ecoscope.platform.tasks.analysis._time_density import (  # type: ignore[import-untyped]
    AutoScaleGridCellSize,
    CustomGridCellSize,
    TrajectoryAnnotation,
)
from ecoscope.platform.tasks.config._home_range import (  # type: ignore[import-untyped]
    BbmmRasterArgs,
)
from ecoscope.platform.tasks.config._meta_tasks import (  # type: ignore[import-untyped]
    EtdArgsWithOpacity,
)

logger = logging.getLogger(__name__)

OutputDirAnnotation: TypeAlias = Annotated[
    str,
    Field(description="Directory the GeoTIFF will be written to.", exclude=True),
]
FilenameAnnotation: TypeAlias = Annotated[
    str,
    AdvancedField(
        default="etd_raster",
        title="Raster Filename",
        description="Output filename for the ETD raster, without extension.",
    ),
]
BbmmFilenameAnnotation: TypeAlias = Annotated[
    str,
    AdvancedField(
        default="bbmm_raster",
        title="Raster Filename",
        description="Output filename for the BBMM raster, without extension.",
    ),
]
GroupKeyAnnotation: TypeAlias = Annotated[
    Optional[CompositeFilter],
    Field(
        description=(
            "If present (e.g. when fanned out via `map` over a grouped/split trajectory), a hash "
            "of the group key is prepended to `filename` so groups don't overwrite each other's "
            "GeoTIFF, matching the same per-group hash used by every other output file and the "
            "dashboard's `views_json` keys."
        ),
        exclude=True,
    ),
]


def _hash_grouper_key(group_key: CompositeFilter) -> str:
    json_key = {cond[0]: cond[2] for cond in group_key}
    return hashlib.sha256(json.dumps(json_key, sort_keys=True).encode()).hexdigest()[:6]


def _filename_prefix_from_group_key(group_key: CompositeFilter | None) -> str | None:
    return _hash_grouper_key(group_key) if group_key else None


def _remove_file_scheme(path: str) -> str:
    """Remove a file:// scheme prefix from a path if present."""
    if not path.startswith("file://"):
        return path

    parsed = urlparse(path)

    if parsed.scheme == "file" and parsed.path:
        path = parsed.path
    elif parsed.scheme == "file":
        path = parsed.netloc

    if os.name == "nt":
        # Remove leading slash before drive letter: /C:/path -> C:/path
        if path.startswith("/") and len(path) > 2 and path[2] in (":", "|"):
            path = path[1:]

        path = path.replace("/", "\\")
        path = path.replace("|", ":")

    return path


def _build_output_path(output_dir: str, filename: str, group_key: CompositeFilter | None) -> str:
    output_dir = _remove_file_scheme(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    prefix = _filename_prefix_from_group_key(group_key)
    full_filename = f"{prefix}_{filename}.tif" if prefix else f"{filename}.tif"
    return os.path.join(output_dir, full_filename)


def export_geotiff(
    raster_data: RasterData,
    output_path: str,
    band_count: int = 1,
    dtype: str = "float32",
    nodata: float | str = "nan",
) -> None:
    """Write a utilization-distribution surface to disk as a GeoTIFF.

    Cells with no computed density (never touched by any segment's window,
    or exactly zero) are masked to `nodata` so they render as transparent
    background in GIS tools, not a solid zero fill.
    """
    from ecoscope.io.raster import RasterPy  # type: ignore[import-untyped]

    nodata_value = float("nan") if nodata == "nan" else nodata

    ndarray = raster_data.data.copy()
    ndarray[np.isnan(ndarray) | (ndarray == 0)] = nodata_value

    rows, columns = ndarray.shape
    RasterPy.write(
        ndarray,
        fp=output_path,
        columns=columns,
        rows=rows,
        band_count=band_count,
        dtype=dtype,
        crs=raster_data.crs,
        transform=raster_data.transform,
        nodata=nodata_value,
    )


@register()
def generate_etd_raster(
    trajectory_gdf: TrajectoryAnnotation,
    combined_params: EtdArgsWithOpacity,
    output_dir: OutputDirAnnotation,
    filename: FilenameAnnotation = "etd_raster",
    group_key: GroupKeyAnnotation = None,
) -> Annotated[str, Field(description="Path to the written ETD utilization-distribution GeoTIFF.")]:
    """Compute the Elliptical Time-Density (ETD) utilization distribution raster and persist it as a GeoTIFF.

    Takes the same `EtdArgsWithOpacity` combined-params object as `call_etd_from_combined_params`
    (see `set_etd_args_with_opacity`), so the raster and the percentile table are always built from
    the same grid settings. Mirrors the grid-sizing behaviour of `calculate_elliptical_time_density`,
    but writes the underlying raster surface to disk instead of returning percentile-area polygons.
    """
    from ecoscope.analysis.UD import (  # type: ignore[import-untyped]
        calculate_etd_range,
        grid_size_from_geographic_extent,
    )
    from ecoscope.io.raster import RasterProfile  # type: ignore[import-untyped]

    if trajectory_gdf is None or trajectory_gdf.empty:
        raise ValueError("generate_etd_raster: `trajectory_gdf` is empty.")

    etd_params = combined_params.get_etd_params()
    auto_scale_or_custom_cell_size = etd_params["auto_scale_or_custom_cell_size"] or AutoScaleGridCellSize()

    if isinstance(auto_scale_or_custom_cell_size, CustomGridCellSize):
        pixel_size = auto_scale_or_custom_cell_size.grid_cell_size
    else:
        pixel_size = grid_size_from_geographic_extent(trajectory_gdf, scale_factor=500)

    raster_profile = RasterProfile(
        pixel_size=pixel_size,
        crs=etd_params["crs"],
        nodata_value=etd_params["nodata_value"],
        band_count=etd_params["band_count"],
    )

    trajectory_gdf = trajectory_gdf.sort_values("segment_start")
    output_path = _build_output_path(output_dir, filename, group_key)

    raster_data = calculate_etd_range(
        trajectory=trajectory_gdf,
        max_speed_kmhr=etd_params["max_speed_factor"] * trajectory_gdf["speed_kmhr"].max(),
        raster_profile=raster_profile,
        expansion_factor=etd_params["expansion_factor"],
    )

    if raster_data is None or raster_data.data is None or raster_data.data.size == 0:
        raise ValueError(
            "generate_etd_raster: no raster data was generated - the trajectory extent may be too small "
            "relative to the configured grid cell size."
        )

    # dtype="float64" (not export_geotiff's own "float32" default) matches this
    # task's original written-file precision exactly, from before it shared
    # export_geotiff with generate_bbmm_raster.
    export_geotiff(
        raster_data,
        output_path,
        band_count=etd_params["band_count"],
        dtype="float64",
        nodata=etd_params["nodata_value"],
    )

    logger.info(f"ETD raster written to: {output_path}")
    return output_path


@register()
def generate_bbmm_raster(
    trajectory_gdf: TrajectoryAnnotation,
    combined_params: BbmmRasterArgs,
    output_dir: OutputDirAnnotation,
    filename: BbmmFilenameAnnotation = "bbmm_raster",
    group_key: GroupKeyAnnotation = None,
) -> Annotated[str, Field(description="Path to the written BBMM utilization-distribution GeoTIFF.")]:
    """Compute the Brownian Bridge Movement Model (BBMM) utilization
    distribution raster and persist it as a GeoTIFF.

    Takes the same `BbmmRasterArgs` combined-params shape
    `get_bbmm_raster_params_from_args` derives from `BbmmMethodArgs`, so the
    raster and the percentile table are always built from the same grid
    settings. Re-runs the calculation rather than reusing a prior result
    (the percentile task never returns the raw raster). nodata/band_count
    aren't user-configurable for BBMM (unlike ETD), so `export_geotiff` is
    called with its own defaults.
    """
    from ecoscope.analysis.UD import (
        calculate_bbmm_range,  # type: ignore[import-untyped]
    )

    if trajectory_gdf is None or trajectory_gdf.empty:
        raise ValueError("generate_bbmm_raster: `trajectory_gdf` is empty.")

    raster_data = calculate_bbmm_range(trajectory_gdf, **combined_params.get_bbmm_params())

    if raster_data is None or raster_data.data is None or raster_data.data.size == 0:
        raise ValueError(
            "generate_bbmm_raster: no raster data was generated - the trajectory extent may be too small "
            "relative to the estimated grid cell size."
        )

    output_path = _build_output_path(output_dir, filename, group_key)
    export_geotiff(raster_data, output_path)

    logger.info(f"BBMM raster written to: {output_path}")
    return output_path
