"""NDVI map tile task.

Lives in ``analysis`` (like ``create_forest_layers``) rather than ``io`` because
importing the pydeck result types from ``io`` creates a circular import.

Moved from ``ecoscope-workflows-ext-custom`` so that the NDVI workflow can rely on a
single task library, and extended with the same optional water mask as
``calculate_ndvi_range`` so that the NDVI map and the NDVI trend exclude the same
pixels.
"""

import hashlib
from typing import Annotated, Literal

from pydantic import Field
from pydantic.json_schema import SkipJsonSchema
from wt_registry import register

from ecoscope.platform.annotations import AdvancedField, AnyGeoDataFrame
from ecoscope.platform.connections import EarthEngineClient
from ecoscope.platform.serde import _persist_bytes
from ecoscope.platform.tasks.filter._filter import TimeRange
from ecoscope.platform.tasks.io._earthengine import (
    _DETAILED_QA_BAND,
    _MODIS_CALCULATED,
    _MODIS_PRECOMPUTED,
    _NIR_BAND,
    _RED_BAND,
    NDVIMethod,
    _land_mask,
)
from ecoscope.platform.tasks.results._pydeck import (
    BitmapLayerDefinition,
    LegendSegment,
    LegendValue,
)

_DEFAULT_PALETTE = ["#A0522D", "#C4A882", "#D4CC6A", "#8FBC6B", "#5A8C4A", "#2E5E2E"]


@register(tags=["io"])
def create_ndvi_tile(
    client: Annotated[
        EarthEngineClient,
        Field(
            description="A named Google EarthEngine connection.",
            exclude=True,
        ),
    ],
    roi: Annotated[
        AnyGeoDataFrame,
        Field(
            description="Region of interest to clip the raster to.",
            exclude=True,
        ),
    ],
    time_range: Annotated[
        TimeRange,
        Field(
            description="Time range to filter the image collection.",
            exclude=True,
        ),
    ],
    ndvi_method: Annotated[
        NDVIMethod,
        Field(
            description="Method to obtain NDVI values. "
            "'MODIS MYD13A1 16-Day Composite': Uses pre-calculated NDVI from 16-day composites. "
            "'MODIS MCD43A4 Daily NBAR': Computes NDVI from NIR/Red bands.",
        ),
    ] = "MODIS MYD13A1 16-Day Composite",
    reducer: Annotated[
        Literal["mean", "median", "min", "max"],
        Field(
            description="Reducer to apply to the ImageCollection over the time range.",
        ),
    ] = "mean",
    palette: Annotated[
        list[str] | SkipJsonSchema[None],
        AdvancedField(
            default=None,
            description="Color palette for visualization, as hex color strings.",
        ),
    ] = None,
    scale: Annotated[
        int,
        AdvancedField(
            description="Scale in meters for computing min/max statistics.",
            default=500,
        ),
    ] = 500,
    root_path: Annotated[
        str | SkipJsonSchema[None],
        Field(
            description="Root path for persisting GeoTIFF. If None, GeoTIFF export is skipped.",
        ),
    ] = None,
    filename_prefix: Annotated[
        str | SkipJsonSchema[None],
        AdvancedField(
            description="Optional filename prefix for the GeoTIFF output.",
            default=None,
        ),
    ] = None,
    opacity: Annotated[
        float,
        Field(
            description="Opacity of the NDVI tile layer (0.0 to 1.0).",
            ge=0.0,
            le=1.0,
        ),
    ] = 1.0,
    mask_water: Annotated[
        bool,
        Field(
            description="Exclude water bodies (e.g. lakes) from the NDVI map. "
            "'MODIS MYD13A1 16-Day Composite' uses the product's DetailedQA land/water flag; "
            "'MODIS MCD43A4 Daily NBAR' uses the static MODIS MOD44W land/water mask. "
            "Masks the same pixels as calculate_ndvi_range.",
        ),
    ] = False,
) -> Annotated[BitmapLayerDefinition, Field()]:
    """Creates an NDVI tile layer from MODIS Earth Engine ImageCollections.

    Supports the same two methods as calculate_ndvi_range:
        - MODIS MYD13A1 16-Day Composite: Pre-calculated NDVI band, scaled by 0.0001
        - MODIS MCD43A4 Daily NBAR: Computes NDVI as (NIR - Red) / (NIR + Red)

    When ``mask_water`` is True, non-land pixels are masked on every image before the
    temporal reduction, so they are transparent on the map, excluded from the legend
    range, and excluded from the GeoTIFF export.

    Returns a BitmapLayerDefinition with an embedded base64 PNG and an auto-scaled legend.
    """
    import base64

    import ee
    import requests

    if not palette:
        palette = _DEFAULT_PALETTE

    # Build EE geometry from ROI
    roi_gdf = roi.to_crs("EPSG:4326")  # type: ignore[operator]
    roi_geometry = roi_gdf.dissolve().geometry.iloc[0]
    ee_geometry = ee.Geometry(roi_geometry.__geo_interface__)

    reducer_map = {
        "mean": ee.Reducer.mean(),
        "median": ee.Reducer.median(),
        "min": ee.Reducer.min(),
        "max": ee.Reducer.max(),
    }

    date_start = time_range.since.isoformat()
    date_end = time_range.until.isoformat()

    if ndvi_method == "MODIS MYD13A1 16-Day Composite":

        def scale_ndvi(i):
            ndvi = i.select("NDVI").multiply(0.0001)
            if mask_water:
                ndvi = ndvi.updateMask(_land_mask(i, ndvi_method))
            return ndvi.set("system:time_start", i.get("system:time_start"))

        img = (
            ee.ImageCollection(_MODIS_PRECOMPUTED)
            .select(["NDVI", _DETAILED_QA_BAND] if mask_water else ["NDVI"])
            .filterDate(date_start, date_end)
            .map(scale_ndvi)
            .reduce(reducer_map[reducer])
            .clip(ee_geometry)
        )
    else:  # calculated

        def compute_ndvi(i):
            nir = i.select(_NIR_BAND)
            red = i.select(_RED_BAND)
            ndvi = nir.subtract(red).divide(nir.add(red)).rename("NDVI")
            if mask_water:
                ndvi = ndvi.updateMask(_land_mask(i, ndvi_method))
            return ndvi.set("system:time_start", i.get("system:time_start"))

        img = (
            ee.ImageCollection(_MODIS_CALCULATED)
            .select([_NIR_BAND, _RED_BAND])
            .filterDate(date_start, date_end)
            .map(compute_ndvi)
            .reduce(reducer_map[reducer])
            .clip(ee_geometry)
        )

    # Compute min/max for visualization (values should be in 0-1 range)
    stats = img.reduceRegion(
        reducer=ee.Reducer.minMax(),
        geometry=ee_geometry,
        bestEffort=True,
        scale=scale,
    ).getInfo()

    stat_values = [v for v in stats.values() if v is not None]
    img_min = min(stat_values) if stat_values else 0
    img_max = max(stat_values) if stat_values else 1

    # Download image as PNG and embed as base64 data URI.
    # This avoids the GEE tile URL auth token expiration (401 errors in saved HTML).

    bounds = list(roi_gdf.total_bounds)  # [minx, miny, maxx, maxy] in EPSG:4326

    thumb_url = ee.Image(img).getThumbURL(
        {
            "region": ee_geometry,
            "dimensions": "1024x1024",
            "format": "png",
            "min": img_min,
            "max": img_max,
            "palette": palette,
        }
    )
    response = requests.get(thumb_url)
    response.raise_for_status()

    data_uri = "data:image/png;base64," + base64.b64encode(response.content).decode()

    # Build legend segment
    n = len(palette)
    legend_values: list[LegendValue] = []
    if n == 1:
        legend_values = [LegendValue(label=f"{img_min:.2f}", color=palette[0])]
    else:
        for i, color in enumerate(palette):
            value = img_min + (img_max - img_min) * i / (n - 1)
            legend_values.append(LegendValue(label=f"{value:.2f}", color=color))

    legend = LegendSegment(
        values=legend_values,
        title="NDVI",
    )

    # Export GeoTIFF if root_path is provided
    if root_path is not None:
        download_url = img.getDownloadURL(
            {
                "region": ee_geometry,
                "scale": scale,
                "format": "GEO_TIFF",
                "crs": "EPSG:4326",
            }
        )
        geotiff_response = requests.get(download_url)
        geotiff_response.raise_for_status()
        prefix = filename_prefix or "ndvi"
        fname = f"{prefix}_{hashlib.sha256(geotiff_response.content).hexdigest()[:7]}.tif"
        _persist_bytes(geotiff_response.content, root_path, fname)

    return BitmapLayerDefinition(image=data_uri, bounds=bounds, opacity=opacity, legend=legend)
