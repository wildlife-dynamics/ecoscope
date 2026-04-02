from datetime import datetime, timezone
from typing import Annotated, Literal, Optional, cast

from pydantic import Field
from wt_registry import register

from ecoscope.platform.annotations import AnyDataFrame, AnyGeoDataFrame
from ecoscope.platform.connections import EarthEngineClient
from ecoscope.platform.tasks.filter._filter import BaselineTimeRange, TimeRange

NDVIMethod = Literal["MODIS MYD13A1 16-Day Composite", "MODIS MCD43A4 Daily NBAR"]
GroupingUnit = Literal["month", "week", "day_of_year", "modis_16_day"]

# Constants for MODIS collections
_MODIS_PRECOMPUTED = "MODIS/061/MYD13A1"
_MODIS_CALCULATED = "MODIS/061/MCD43A4"
_NIR_BAND = "Nadir_Reflectance_Band2"
_RED_BAND = "Nadir_Reflectance_Band1"

# MODIS data availability start date (Feb 2000)
_MODIS_DATA_START = datetime(2000, 2, 1, tzinfo=timezone.utc)

# Analysis scale in meters (matches MODIS 500m resolution)
_ANALYSIS_SCALE = 500.0


def _validate_modis_start_date(time_range: TimeRange) -> TimeRange:
    """Validate that time range doesn't start before MODIS data availability."""
    if time_range.since < _MODIS_DATA_START:
        raise ValueError(
            f"Historical time range cannot start before {_MODIS_DATA_START.date()} "
            f"(MODIS data availability). Got: {time_range.since.date()}"
        )
    return time_range


@register(tags=["io"])
def calculate_ndvi_range(
    client: EarthEngineClient,
    roi: AnyGeoDataFrame,
    time_range: Annotated[TimeRange, Field(description="Current time range for trend analysis")],
    image_size: Annotated[
        int,
        Field(
            description="Number of historical satellite images to fetch. "
            "Set to a large value to retrieve all available historical data. "
            "Only used when baseline_time_range is not provided."
        ),
    ] = 1_000_000_000,
    baseline_time_range: Annotated[
        Optional[BaselineTimeRange],
        Field(
            description="Explicit historical baseline and current time range. "
            "When provided, overrides time_range and image_size."
        ),
    ] = None,
    ndvi_method: Annotated[
        NDVIMethod,
        Field(
            description="Method to obtain NDVI values. "
            "'MODIS MYD13A1 16-Day Composite': Uses pre-calculated NDVI from 16-day composites. "
            "Provides quality-filtered 'best pixel' values at 500m resolution with ~0.025 accuracy. "
            "Better for phenology studies but may saturate in dense canopies. "
            "'MODIS MCD43A4 Daily NBAR': Uses daily nadir BRDF-adjusted reflectance. "
            "Computes NDVI from NIR/Red bands with view-angle correction for consistent measurements. "
            "Higher temporal resolution but more susceptible to cloud gaps."
        ),
    ] = "MODIS MYD13A1 16-Day Composite",
    grouping_unit: Annotated[
        GroupingUnit,
        Field(
            description="Temporal unit for grouping historical data when calculating statistics. "
            "'month': Compare against same calendar month (1-12). "
            "'week': Compare against same ISO week number (1-53). "
            "'day_of_year': Compare against same day of year (1-366). "
            "'modis_16_day': Compare against same MODIS 16-day composite period (0-22)."
        ),
    ] = "month",
) -> AnyDataFrame:
    """Calculate NDVI values over an ROI using Earth Engine Image Collections.

    Compares current NDVI values against historical statistics (min, max, mean)
    grouped by a configurable temporal unit to identify trends and anomalies.

    Supports two modes:
        - Mode A (default): Uses time_range + image_size to fetch N historical images
          before the current period. Deprecated; will be removed in a future release.
        - Mode B: Uses baseline_time_range with explicit historical_start, current_start,
          current_end for precise control over both periods.

    Methods:
        - MODIS MYD13A1 16-Day Composite: Uses pre-calculated NDVI band
        - MODIS MCD43A4 Daily NBAR: Computes NDVI from (NIR - Red) / (NIR + Red)

    Grouping:
        - month: Compare against same calendar month across historical years
        - week: Compare against same ISO week number across historical years
        - day_of_year: Compare against same day of year across historical years
        - modis_16_day: Compare against same MODIS 16-day composite period (0-22) across historical years
    """
    import ee
    import pandas as pd

    from ecoscope.io import eetools  # type: ignore[import-untyped]

    if baseline_time_range is not None:
        # Mode B: explicit three-date configuration
        filter_start = baseline_time_range.historical_start.isoformat()
        filter_end = baseline_time_range.current_end.isoformat()
        current_since = baseline_time_range.current_start
        current_until = baseline_time_range.current_end
        historical_since = baseline_time_range.historical_start
        historical_until = baseline_time_range.current_start
        n_images = None  # fetch all images in the filtered range
    else:
        # Mode A: time_range + image_size
        filter_start = _MODIS_DATA_START.isoformat()
        filter_end = time_range.until.isoformat()
        current_since = time_range.since
        current_until = time_range.until
        historical_since = _MODIS_DATA_START
        historical_until = time_range.since
        n_images = image_size

    if ndvi_method == "MODIS MYD13A1 16-Day Composite":
        img_coll = (
            ee.ImageCollection(_MODIS_PRECOMPUTED)
            .filterDate(filter_start, filter_end)
            .select("NDVI")
            .map(
                lambda img: img.multiply(0.0001)
                .set("system:time_start", img.get("system:time_start"))
                .set("id", img.get("id"))
            )
            .sort("system:time_start")
        )
    else:  # calculated
        img_coll = (
            ee.ImageCollection(_MODIS_CALCULATED)
            .filterDate(filter_start, filter_end)
            .select([_NIR_BAND, _RED_BAND])
            .sort("system:time_start")
        )

    # Avoid mutating input - work with a copy
    roi_with_time = roi.copy()
    roi_with_time["tmp_time"] = current_until

    n_before = n_images if n_images is not None else img_coll.size().getInfo()

    ee_data = eetools.label_gdf_with_temporal_image_collection_by_feature(
        gdf=roi_with_time,
        time_col_name="tmp_time",
        n_before=n_before,
        n_after=0,
        img_coll=img_coll,
        region_reducer=ee.Reducer.mean(),
        scale=_ANALYSIS_SCALE,
    )

    ee_data = ee_data.reset_index()
    ee_data["img_date"] = pd.to_datetime(ee_data["img_date"]).dt.tz_localize("UTC")

    if ndvi_method == "MODIS MYD13A1 16-Day Composite":
        assert len(ee_data.columns) == 3, f"Expected 3 columns, got {len(ee_data.columns)}"
        ee_data.columns = ["name", "img_date", "NDVI"]
    else:  # calculated
        ee_data["NDVI"] = (ee_data[_NIR_BAND] - ee_data[_RED_BAND]) / (ee_data[_NIR_BAND] + ee_data[_RED_BAND])
        idx_col = ee_data.columns[0]
        ee_data = ee_data[[idx_col, "img_date", "NDVI"]]
        ee_data.columns = ["name", "img_date", "NDVI"]

    cur_data = ee_data[(ee_data.img_date >= current_since) & (ee_data.img_date <= current_until)]
    historical_data = ee_data[(ee_data.img_date >= historical_since) & (ee_data.img_date < historical_until)]

    def get_grouping_key(dt_series, unit: str):
        """Map a datetime or datetime accessor to a grouping key for the given unit."""
        if unit == "month":
            return dt_series.month
        elif unit == "week":
            return dt_series.isocalendar().week
        elif unit == "modis_16_day":
            return (dt_series.dayofyear - 1) // 16
        else:  # day_of_year
            return dt_series.dayofyear

    def calc_mean_range(row):
        current_key = get_grouping_key(row.img_date, grouping_unit)
        grouped_hist = historical_data[get_grouping_key(historical_data.img_date.dt, grouping_unit) == current_key]
        return pd.Series(
            {
                "min": grouped_hist["NDVI"].min(),
                "max": grouped_hist["NDVI"].max(),
                "mean": grouped_hist["NDVI"].mean(),
                "NDVI": row["NDVI"],
                "img_date": row.img_date,
            }
        )

    if cur_data.empty:
        return cast(
            AnyDataFrame,
            pd.DataFrame(columns=["min", "max", "mean", "NDVI", "img_date"]),
        )

    return cur_data.apply(calc_mean_range, axis=1).reset_index(drop=True)


@register(tags=["io"])
def determine_season_windows(
    client: EarthEngineClient,
    roi: AnyGeoDataFrame,
    time_range: Annotated[TimeRange, Field(description="Time range filter")],
) -> AnyDataFrame:
    """Determine wet/dry season windows from NDVI values over an ROI.

    Computes standardized NDVI values using MODIS MCD43A4 daily NBAR data,
    then identifies seasonal transition points to classify time windows
    as "dry" or "wet" seasons.
    """
    import pandas as pd

    from ecoscope.analysis.seasons import (  # type: ignore[import-untyped]
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

    ndvi_vals = pd.concat(
        [
            std_ndvi_vals(
                img_coll=_MODIS_CALCULATED,
                nir_band=_NIR_BAND,
                red_band=_RED_BAND,
                aoi=merged_roi,
                start=date_chunks[t - 1],
                end=date_chunks[t],
            )
            for t in range(1, len(date_chunks))
        ]
    )

    # Calculate the seasonal transition point
    cuts = val_cuts(ndvi_vals, 2)

    # Determine the seasonal time windows
    return cast(AnyDataFrame, seasonal_windows(ndvi_vals, cuts, season_labels=["dry", "wet"]))
