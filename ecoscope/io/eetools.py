import concurrent.futures
import datetime as dt
import functools
import itertools
import json
import logging

import backoff
import ee
import numpy as np
import pandas as pd
import pytz
import tqdm.auto as tqdm

logging.getLogger("urllib3").setLevel(logging.ERROR)  # Filter "Connection pool is full, discarding connection"

logger = logging.getLogger(__name__)

ee_is_initialized = False

ee_initialization_expires = dt.datetime(1970, 1, 1, tzinfo=pytz.utc)
ee_initialization_ttl = dt.timedelta(seconds=86400)


# useful stylizations
colourPalettes = {
    "precipitation": {"palette": ["00FFFF", "0000FF"]},
}


def initialize_earthengine(key_dict):
    """
    This takes a JSON key as a dict.
    :param key_dict:
    :return: a credentials object that can be used to initialize the earth engine library.
    """
    global ee_is_initialized
    global ee_initialization_expires

    try:
        if ee_is_initialized and (dt.datetime.now(tz=pytz.utc) > ee_initialization_expires):
            logger.debug("Earth Engine has already been initialized.")
            return

        sac = ee.ServiceAccountCredentials(key_dict["client_email"], key_data=json.dumps(key_dict))
        ee.Initialize(credentials=sac)

    except Exception as e:
        logger.exception("Not able to initialize EarthEngine: %s", e)
        raise
    else:
        ee_is_initialized = True
        ee_initialization_expires = dt.datetime.now(tz=pytz.utc) + ee_initialization_ttl
        logger.debug("Earth Engine has been initialized.")


def convert_millisecs_datetime(unix_time):
    # Define the UNIX Epoch start date
    epoch = dt.datetime(1970, 1, 1, 0, 0, 0)
    t = epoch + dt.timedelta(milliseconds=unix_time)
    return t


def add_img_time(img):
    """
    A function to add the date range of the image as one of its properties and the start and end values as new bands
    """

    # unable to use the band rename() function here in Python whereas works in JS playground
    img = img.addBands(img.metadata("system:time_start"))
    img = img.addBands(img.metadata("system:time_end"))
    dr = ee.DateRange(ee.Date(img.get("system:time_start")), ee.Date(img.get("system:time_end")))
    return img.set({"date_range": dr})


@backoff.on_exception(
    backoff.expo,
    ee.EEException,
    max_tries=10,
)
def label_gdf_with_img(gdf=None, img=None, region_reducer=None, scale=500.0):
    in_fc = ee.FeatureCollection(gdf.__geo_interface__)
    out_fc = img.reduceRegions(in_fc, region_reducer, scale)
    valid_properties = (
        out_fc.first().propertyNames().filter(ee.Filter.inList("item", in_fc.first().propertyNames()).Not())
    )
    return pd.DataFrame(
        out_fc.select(valid_properties)
        .reduceColumns(ee.Reducer.toList(valid_properties.size()), valid_properties)
        .get("list")
        .getInfo(),
        columns=valid_properties.getInfo(),
        index=gdf.index,
    ).apply(pd.Series.explode)


def _match_gdf_to_img_coll_ids(
    gdf, time_col="", img_coll=None, output_col_name=None, stack_limit_before=1, stack_limit_after=1
):
    """
    A function that will add a column to a gdf (output_col_name) that contains
    the stack_limit_before -> stack_limit_after temporally closest image IDs from an image collection.
    :param gdf:
    :param time_col:
    :param img_coll:
    :param output_col_name:
    :param stack_limit_before:
    :param stack_limit_after:
    :return: None
    """

    try:
        # Step 1: download the img_coll image times and ids to a dataframe
        logger.info("Downloading Image Collection IDs and Dates")
        img_data = (
            img_coll.reduceColumns(ee.Reducer.toList(2), ["system:index", "system:time_start"]).get("list").getInfo()
        )
        img_data = np.array(img_data)
        img_data = (
            pd.DataFrame(
                {"img_id": img_data[:, 0], "img_date": pd.to_datetime(img_data[:, 1], unit="ms").tz_localize("UTC")}
            )
            .sort_values("img_date")
            .set_index("img_date")
        )

        # Step 2: determine the closest image IDs to a given feature date
        def determine_img_ids(row):
            row_time = row.get(time_col, pd.Timestamp(0, tz="utc"))
            nearest_index = img_data.index.get_loc(row_time, method="nearest")
            lower = int(nearest_index - stack_limit_before)
            if lower < 0:
                lower = 0
            upper = int(nearest_index + stack_limit_after)
            if upper > len(img_data.index) - 1:
                upper = len(img_data.index)
            img_ids = img_data.iloc[
                lower:upper,
            ]["img_id"].to_list()
            return img_ids

        logger.info("Matching Features to Image IDs")
        gdf[output_col_name] = gdf.apply(determine_img_ids, axis=1)

    except Exception as e:
        logger.error(str(e))


@backoff.on_exception(
    backoff.expo,
    ee.EEException,
    max_tries=10,
)
def label_gdf_with_temporal_image_collection_by_feature(
    gdf=None,
    time_col_name=None,
    stack_limit_before=1,
    stack_limit_after=1,
    img_coll=None,
    region_reducer=None,
    scale=500.0,
):

    # Match the features to the necessary image collection images
    _match_gdf_to_img_coll_ids(
        gdf=gdf,
        time_col=time_col_name,
        img_coll=img_coll,
        output_col_name="img_ids",
        stack_limit_before=stack_limit_before,
        stack_limit_after=stack_limit_after,
    )

    in_fc = ee.FeatureCollection(gdf[["geometry", "img_ids"]].__geo_interface__)

    def feat_func(feat):
        tmp_coll = img_coll.filter(ee.Filter.inList("system:index", feat.get("img_ids")))

        def region_reduc(img):
            return img.set(
                {
                    "img_date": img.date().format(),
                    "img_vals": img.reduceRegion(reducer=region_reducer, geometry=feat.geometry(), scale=scale),
                }
            )

        return feat.set(
            "values",
            tmp_coll.map(region_reduc).reduceColumns(ee.Reducer.toList(2), ["img_date", "img_vals"]).values().get(0),
        )

    logger.info("Labeling Features with Image Collection Values")
    out_fc = in_fc.map(feat_func, True)
    result = pd.DataFrame(
        out_fc.select("values").reduceColumns(ee.Reducer.toList(1), ["values"]).get("list").getInfo(),
        columns=["values"],
        index=gdf.index,
    ).apply(pd.Series.explode)
    result["img_date"] = result["values"].str[0]
    result["values"] = result["values"].str[1]
    result = pd.concat([result.drop(columns=["values"]), result["values"].apply(pd.Series)], ignore_index=False, axis=1)
    return result


@backoff.on_exception(
    backoff.expo,
    ee.EEException,
    max_tries=10,
)
def label_gdf_with_temporal_image_collection_by_timespan(
    gdf=None,
    img_coll=None,
    image_radius=0,
    add_time=False,
    region_reducer="toList",
    df_chunk_size=25000,
    max_workers=1,
):
    img_list = img_coll.toList(img_coll.size())

    # @TODO sort. Current code assumes img_coll is sorted by `system:time_start`
    times = np.array(img_coll.aggregate_array("system:time_start").getInfo())
    time_bins = pd.to_datetime(
        np.concatenate(([0], (times[:-1] + times[1:]) / 2, [pd.Timestamp.max.timestamp() * 1000 - 1])),
        unit="ms",
        utc=True,
    )

    max_idx = len(times)
    imgs, chunks = zip(
        *itertools.chain(
            *[
                [
                    (
                        ee.ImageCollection(
                            img_list.slice(max(idx - image_radius, 0), min(idx + image_radius + 1, max_idx))
                        ).toBands(),
                        gs.iloc[i : i + df_chunk_size],
                    )
                    for i in range(0, len(gs), df_chunk_size)
                ]
                for idx, gs in gdf.geometry.groupby(pd.cut(gdf.fixtime, time_bins, labels=False))
            ]
        )
    )

    def f(img, chunk):
        return label_gdf_with_img(chunk.geometry, img, region_reducer).melt(ignore_index=False)

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(tqdm.tqdm(executor.map(f, imgs, chunks), total=len(chunks)))

    if results:
        results = pd.concat(results)

        if add_time:
            img_names = np.array(img_coll.aggregate_array("system:index").getInfo())

            cat = results.variable.astype("category").cat
            results["time"] = cat.categories.str[: len(img_names[0])].map(
                pd.Series(pd.to_datetime(times, unit="ms", utc=True), index=img_names)
            )[cat.codes]

        return results


def chunk_gdf(
    gdf=None,
    label_func=None,
    label_func_kwargs=None,
    df_chunk_size=25000,
    max_workers=1,
):
    """
    A function that will process the input gdf in chunks and apply the input label_func function over the chunks.

    :param gdf:
        a geopandas dataframe. The 'geometry' column can be any type pf geometry (point/line/polygon). The gdf needs to
        have a column with the name of the image_collection and the column values are lists of the individual image IDs
        that need to be associated with each feature. This step will typically be run with the
        match_img_coll_ids_to_gdf() function beforehand.
    :param label_func: a function to run on the EE cloud that has the signature (feat, kwargs)
    :param label_func_kwargs: a dictionary of parameters to provide to the label_func
    :param df_chunk_size: how many features (rows) to process at once within EE
    :param max_workers: the number of chunks to process concurrently
    :return: a dataframe with the same index as the input gdf and where each pixel value (or reduced value) is a row
    """
    chunks = [gdf.iloc[i : i + df_chunk_size].copy() for i in range(0, len(gdf), df_chunk_size)]

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(
            tqdm.tqdm(executor.map(functools.partial(label_func, **label_func_kwargs), chunks), total=len(chunks))
        )

    if results:
        return pd.concat(results)


def calculate_anomaly(
    gdf=None,
    img_coll=None,
    historical_start="2000-01-01",
    start="2010-01-01",
    end="2022-01-01",
    scale=5000.0,
):
    """
    Compute anomalies by subtracting the historical_start mean from each image in a collection of start->end images.
    :param gdf: the input geodataframe
    :param img_coll: the input EE image collection
    :param historical_start: start time for calculating the mean reference
    :param start: end time for mean reference and the start time for the anomaly calculation
    :param end: end time for the anomaly calculation
    :param scale: the image scale
    :return: a dataframe with same index as input gdf with the img_dates and the anomaly calculation
    """
    fc = ee.FeatureCollection(gdf[["geometry"]].__geo_interface__)

    mean_ref = img_coll.filterDate(historical_start, start).mean()

    img_coll = img_coll.filterDate(start, end).sort("system:time_start", False)

    def feat_func(feat):
        def img_func(img):
            return img.set(
                {
                    "img_date": img.date().format(),
                    "mean": img.subtract(mean_ref).reduceRegion("mean", feat.geometry(), scale),
                }
            )

        return feat.set(
            "values",
            img_coll.map(img_func, True).reduceColumns(ee.Reducer.toList(2), ["img_date", "mean"]).values().get(0),
        )

    logger.info("Calculating anomaly")

    vals = fc.map(feat_func, True)
    result = pd.DataFrame(
        vals.select("values").reduceColumns(ee.Reducer.toList(1), ["values"]).get("list").getInfo(),
        columns=["values"],
        index=gdf.index,
    ).apply(pd.Series.explode)

    result["img_date"] = result["values"].str[0]
    result["values"] = result["values"].str[1]
    result.sort_values(by="img_date", inplace=True)
    result = pd.concat(
        [result.drop(columns=["values"]), result["values"].apply(pd.Series).cumsum()], ignore_index=False, axis=1
    )

    return result
