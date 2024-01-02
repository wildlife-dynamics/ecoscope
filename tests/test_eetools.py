import ee
import pandas as pd
import pytest

import ecoscope

if not pytest.earthengine:
    pytest.skip("Skipping tests because connection to Earth Engine is not available.", allow_module_level=True)


def test_albedo_anomaly(aoi_gdf):
    tmp_gdf = aoi_gdf.to_crs(4326)

    params = {
        "img_coll": ee.ImageCollection("MODIS/006/MCD43C3").select("Albedo_BSA_vis"),
        "historical_start": "2000-01-01",
        "start": "2010-01-01",
        "end": "2022-01-01",
        "scale": 5000.0,
    }

    result = ecoscope.io.eetools.chunk_gdf(
        gdf=tmp_gdf,
        label_func=ecoscope.io.eetools.calculate_anomaly,
        label_func_kwargs=params,
        df_chunk_size=1,
        max_workers=5,
    )

    assert isinstance(result, pd.DataFrame)
    assert result["Albedo_BSA_vis"].mean() > 0


def test_label_gdf_with_temporal_image_collection_by_features_aois(aoi_gdf):
    aoi_gdf = aoi_gdf.to_crs(4326)

    # Add a time_column to the gdf
    aoi_gdf["time"] = pd.Timestamp.utcnow() - pd.Timedelta(days=365)

    img_coll = (
        ee.ImageCollection("MODIS/061/MYD13A1")
        .select("NDVI")
        .map(
            lambda img: img.multiply(0.0001)
            .set("system:time_start", img.get("system:time_start"))
            .set("id", img.get("id"))
        )
        .sort("system:time_start")
    )

    params = {
        "time_col_name": "time",
        "stack_limit_before": 10,
        "stack_limit_after": 10,
        "img_coll": img_coll,
        "region_reducer": "mean",
        "scale": 500.0,
    }

    results = ecoscope.io.eetools.chunk_gdf(
        gdf=aoi_gdf,
        label_func=ecoscope.io.eetools.label_gdf_with_temporal_image_collection_by_feature,
        label_func_kwargs=params,
        df_chunk_size=10,
        max_workers=1,
    )

    assert results["NDVI"].explode().mean() > 0


def test_label_gdf_with_temporal_image_collection_by_features_relocations(movbank_relocations):
    tmp_gdf = movbank_relocations[["fixtime", "geometry"]].iloc[0:1000]

    img_coll = ee.ImageCollection("MODIS/MCD43A4_006_NDVI").select("NDVI")  # Daily NDVI images

    params = {
        "time_col_name": "fixtime",
        "stack_limit_before": 1,
        "stack_limit_after": 1,
        "img_coll": img_coll,
        "region_reducer": "toList",
        "scale": 1.0,
    }

    results = ecoscope.io.eetools.chunk_gdf(
        gdf=tmp_gdf,
        label_func=ecoscope.io.eetools.label_gdf_with_temporal_image_collection_by_feature,
        label_func_kwargs=params,
        df_chunk_size=25000,
        max_workers=1,
    )

    assert results["NDVI"].explode().mean() > 0


def test_label_gdf_with_img(movbank_relocations):
    tmp_gdf = movbank_relocations[["geometry"]]
    tmp_gdf = tmp_gdf[0:1000]

    img = ee.Image("USGS/SRTMGL1_003").select("elevation")

    params = {"img": img, "region_reducer": "toList", "scale": 1.0}

    results = ecoscope.io.eetools.chunk_gdf(
        gdf=tmp_gdf,
        label_func=ecoscope.io.eetools.label_gdf_with_img,
        label_func_kwargs=params,
        df_chunk_size=20000,
        max_workers=1,
    )

    assert results["list"].mean() > 0
