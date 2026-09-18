import logging
from typing import Annotated, cast

import pandas as pd
from pydantic import Field
from wt_registry import register

from ecoscope.platform.annotations import AdvancedField, AnyDataFrame, AnyGeoDataFrame
from ecoscope.platform.connections import EarthEngineClient
from ecoscope.platform.tasks.filter._filter import TimeRange
from ecoscope.platform.tasks.results._pydeck import (
    BitmapLayerDefinition,
    LegendSegment,
    LegendValue,
)
from ecoscope.platform.tasks.transformation._crs import ensure_wgs84

logger = logging.getLogger(__name__)

_ACRES_PER_SQUARE_METER = 0.000247105


def _parse_year_range(time_range: TimeRange | None) -> tuple[int | None, int | None]:
    """Return (start_year, end_year) from a TimeRange, or (None, None) if not provided."""
    if time_range is None:
        return None, None
    return time_range.since.year, time_range.until.year


def _make_treecover_mask(gfc, tree_cover_threshold: float):
    """Return (cover_img, cover_mask) from a Hansen Global Forest Change image.

    cover_img - binary pixel image (values -> 1) masked to the threshold
    cover_mask - the raw boolean mask used to filter loss pixels
    """
    cover = gfc.select(["treecover2000"])
    mask = cover.gte(tree_cover_threshold)
    cover = cover.unmask().updateMask(mask)
    return cover.And(cover), mask


@register()
def extract_forest_cover_trends(
    client: EarthEngineClient,
    aoi: Annotated[AnyGeoDataFrame, Field(description="Area of interest geometry")],
    image: Annotated[str, Field(description="Hansen Global Forest Change dataset image name")],
    time_range: Annotated[TimeRange | None, Field(description="Time range for the trend analysis")] = None,
    tree_cover_threshold: Annotated[float, Field(description="Minimum tree cover percentage (0-100)")] = 60.0,
    scale: Annotated[int, AdvancedField(30, description="Pixel scale in meters for reduction")] = 30,
    max_pixels: Annotated[float, AdvancedField(1e9, description="Maximum pixels for reduction")] = 1e9,
) -> AnyDataFrame:
    """Extract yearly forest cover loss/survival trends from the Hansen
    Global Forest Change dataset over `aoi`.

    Returns a DataFrame with columns: year, loss_area, cumsum_loss_area,
    survival_area, cumsum_loss_pct (areas in acres).
    """
    import ee

    aoi = ensure_wgs84(aoi)
    feat_coll = ee.FeatureCollection(aoi.__geo_interface__)
    gfc = ee.Image(image)
    start_year, end_year = _parse_year_range(time_range)

    treecover2000, treecover2000_mask = _make_treecover_mask(gfc, tree_cover_threshold)
    treecover2000_area = treecover2000.multiply(ee.Image.pixelArea()).reduceRegion(
        reducer=ee.Reducer.sum(),
        geometry=feat_coll,
        scale=scale,
        crs="EPSG:3857",
        maxPixels=max_pixels,
    )
    forested_area = (treecover2000_area.getInfo()["treecover2000"] or 0.0) * _ACRES_PER_SQUARE_METER

    loss_year = gfc.select(["lossyear"])
    loss_area_img = gfc.select(["loss"]).updateMask(treecover2000_mask).multiply(ee.Image.pixelArea())
    # Loss is always computed from 2000 (needed for a correct survival_area),
    # but masking out pixels beyond end_year keeps the reduction smaller.
    if end_year:
        loss_area_img = loss_area_img.updateMask(loss_year.lte(max(0, end_year - 2000)))

    loss_by_year = loss_area_img.addBands(loss_year).reduceRegion(
        reducer=ee.Reducer.sum().group(groupField=1),
        geometry=feat_coll,
        scale=scale,
        crs="EPSG:3857",
        maxPixels=max_pixels,
    )

    columns = ["year", "loss_area", "cumsum_loss_area", "survival_area", "cumsum_loss_pct"]
    # If aoi carries a "name" column (e.g. from load_spatial_features_group),
    # propagate it so callers can later concatenate multiple regions' results
    # back into one dataframe and still tell rows apart by region - GAMM's
    # per-site random effects need exactly this to pool across regions.
    if "name" in aoi.columns:
        columns = [*columns, "name"]
    groups = loss_by_year.getInfo()["groups"]
    if not groups:
        return cast(AnyDataFrame, pd.DataFrame(columns=columns))

    forest_survival = pd.DataFrame(groups).rename(columns={"group": "year", "sum": "loss_area"})
    forest_survival["year"] = forest_survival["year"] + 2000
    forest_survival["loss_area"] = forest_survival["loss_area"] * _ACRES_PER_SQUARE_METER
    forest_survival = forest_survival.sort_values("year")
    forest_survival["cumsum_loss_area"] = forest_survival["loss_area"].cumsum()
    forest_survival["survival_area"] = forested_area - forest_survival["cumsum_loss_area"]
    forest_survival["cumsum_loss_pct"] = (
        (forest_survival["cumsum_loss_area"] / forested_area) * 100 if forested_area > 0 else 0.0
    )
    if "name" in aoi.columns:
        forest_survival["name"] = aoi["name"].iloc[0]

    if start_year:
        forest_survival = forest_survival[forest_survival["year"] >= start_year]
    if end_year:
        forest_survival = forest_survival[forest_survival["year"] <= end_year]

    return cast(AnyDataFrame, forest_survival[columns])


@register()
def create_forest_layers(
    client: EarthEngineClient,
    aoi: Annotated[AnyGeoDataFrame, Field(description="Area of interest geometry")],
    image: Annotated[str, Field(description="Hansen Global Forest Change dataset image name")],
    time_range: Annotated[TimeRange | None, Field(description="Time range for the trend analysis")] = None,
    tree_cover_threshold: Annotated[float, Field(description="Minimum tree cover percentage (0-100)")] = 60.0,
    opacity: Annotated[float, AdvancedField(1.0, title="Opacity")] = 1.0,
) -> BitmapLayerDefinition:
    """Render forest cover and forest loss over `time_range` as a single
    bitmap overlay, fetched as a static thumbnail from Earth Engine."""
    import base64

    import ee
    import requests

    roi_gdf = ensure_wgs84(aoi)
    roi_geometry = roi_gdf.dissolve().geometry.iloc[0]  # type: ignore[operator]
    ee_geometry = ee.Geometry(roi_geometry.__geo_interface__)

    gfc = ee.Image(image)
    forest_cover_img, forest_cover_mask = _make_treecover_mask(gfc, tree_cover_threshold)
    start_year, end_year = _parse_year_range(time_range)

    loss_year_img = gfc.select("lossyear")
    mask = loss_year_img.gt(0)
    if start_year:
        mask = mask.And(loss_year_img.gte(start_year - 2000))
    if end_year:
        mask = mask.And(loss_year_img.lte(end_year - 2000))
    forest_loss_img = loss_year_img.updateMask(mask).updateMask(forest_cover_mask)

    cover_vis = forest_cover_img.clip(ee_geometry).visualize(palette=["#236F21"])
    loss_vis = forest_loss_img.clip(ee_geometry).visualize(palette=["#FF0000"])
    blended_img = cover_vis.blend(loss_vis)

    thumb_url = blended_img.getThumbURL({"region": ee_geometry, "dimensions": "1024x1024", "format": "png"})
    response = requests.get(thumb_url)
    response.raise_for_status()
    data_uri = "data:image/png;base64," + base64.b64encode(response.content).decode()

    return BitmapLayerDefinition(
        image=data_uri,
        bounds=list(roi_gdf.total_bounds),
        opacity=opacity,
        legend=LegendSegment(
            title="Forest Change",
            values=[
                LegendValue(label="Forest Cover", color="rgba(35, 111, 33, 1)"),
                LegendValue(label="Forest Loss", color="rgba(255, 0, 0, 1)"),
            ],
        ),
    )
