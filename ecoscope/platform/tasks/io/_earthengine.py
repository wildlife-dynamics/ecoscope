from typing import Annotated, Any, cast

from pydantic import Field
from wt_registry import register

from ecoscope.platform.annotations import AnyDataFrame, AnyGeoDataFrame
from ecoscope.platform.connections import EarthEngineClient
from ecoscope.platform.tasks.filter._filter import TimeRange


@register(tags=["io"])
def calculate_ndvi_range(
    client: EarthEngineClient,
    roi: AnyGeoDataFrame,
    time_range: Annotated[TimeRange, Field(description="Time range filter")],
    img_coll_name: Annotated[str, Field(description="The image collection name")],
    band: Annotated[str, Field(description="The band name")] = "NDVI",
    scale_factor: Annotated[float, Field(description="The scale factor")] = 0.0001,
    analysis_scale: Annotated[float, Field(description="The analysis scale")] = 500.0,
) -> AnyDataFrame:
    """Calculate NDVI values over an ROI using the provided EarthEngine  Image Collection"""

    import ee
    import pandas as pd

    from ecoscope.io import eetools

    img_coll = (
        ee.ImageCollection(img_coll_name)
        .select(band)
        .map(
            lambda img: (
                img.multiply(scale_factor)
                .set("system:time_start", img.get("system:time_start"))
                .set("id", img.get("id"))
            )
        )
        .sort("system:time_start")
    )

    roi["tmp_time"] = time_range.until

    ee_data = eetools.label_gdf_with_temporal_image_collection_by_feature(
        gdf=roi,
        time_col_name="tmp_time",
        n_before=1000000,  # make this big to make sure to get all images
        n_after=0,
        img_coll=img_coll,
        region_reducer=ee.Reducer.mean(),  # will average the NDVI values within the ROI
        scale=analysis_scale,
    )

    ee_data.reset_index(inplace=True)
    ee_data["img_date"] = pd.to_datetime(ee_data["img_date"]).dt.tz_localize("UTC")
    ee_data.columns = ["name", "img_date", "NDVI"]

    cur_data = ee_data[(ee_data.img_date >= time_range.since) & (ee_data.img_date <= time_range.until)]
    historical_data = ee_data[ee_data["img_date"] <= time_range.since]

    def calc_mean_range(x):
        sel = historical_data[(historical_data.img_date.dt.month == x.img_date.month)]
        data = pd.Series(
            {
                "min": sel[band].min(),
                "max": sel[band].max(),
                "mean": sel[band].mean(),
                "NDVI": x[band],
                "img_date": x.img_date,
            }
        )
        return data

    result_df = cur_data.apply(calc_mean_range, axis=1).reset_index(drop=True)

    return cast(
        AnyDataFrame,
        result_df,
    )


@register(tags=["io"])
def determine_season_windows(
    client: EarthEngineClient,
    roi: AnyGeoDataFrame,
    time_range: Annotated[TimeRange, Field(description="Time range filter")],
) -> Any:
    import pandas as pd

    from ecoscope.analysis.seasons import (
        seasonal_windows,
        std_ndvi_vals,
        val_cuts,
    )

    # If there's more than one roi, merge them to one
    merged_roi = roi.to_crs(4326).dissolve().iloc[0]["geometry"]  # type: ignore[operator]

    # Determine wet/dry seasons
    date_chunks = (
        pd.date_range(start=time_range.since, end=time_range.until, periods=5, inclusive="both")
        .to_series()
        .apply(lambda x: x.isoformat())
        .values
    )
    ndvi_vals_list = []
    for t in range(1, len(date_chunks)):
        ndvi_vals_list.append(
            std_ndvi_vals(
                img_coll="MODIS/061/MCD43A4",
                nir_band="Nadir_Reflectance_Band2",
                red_band="Nadir_Reflectance_Band1",
                aoi=merged_roi,
                start=date_chunks[t - 1],
                end=date_chunks[t],
            )
        )
    ndvi_vals = pd.concat(ndvi_vals_list)

    # Calculate the seasonal transition point
    cuts = val_cuts(ndvi_vals, 2)

    # Determine the seasonal time windows
    windows = seasonal_windows(ndvi_vals, cuts, season_labels=["dry", "wet"])

    return windows
