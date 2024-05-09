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
        "n_before": 10,
        "n_after": 10,
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


def test_label_gdf_with_temporal_image_collection_by_features_relocations(movebank_relocations):
    tmp_gdf = movebank_relocations[["fixtime", "geometry"]].iloc[0:1000]

    img_coll = ee.ImageCollection("MODIS/MCD43A4_006_NDVI").select("NDVI")  # Daily NDVI images

    params = {
        "time_col_name": "fixtime",
        "n_before": 1,
        "n_after": 1,
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


def test_label_gdf_with_img(movebank_relocations):
    tmp_gdf = movebank_relocations[["geometry"]]
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


# a subset to ensure we're checking exact match and nearest cases
# includes 3 timestamps, midnight, am, pm
@pytest.fixture
def movebank_relocations_fixed_subset(movebank_relocations):
    return movebank_relocations.loc[329730794:329730795]._append(movebank_relocations.loc[329730810])


@pytest.mark.parametrize(
    "n_before, n_after, output_list",
    [
        (0, 0, [["2009_01_29"], ["2009_01_29"], ["2009_01_30"]]),
        (1, 0, [["2009_01_28", "2009_01_29"], ["2009_01_29"], ["2009_01_29"]]),
        (0, 1, [["2009_01_29", "2009_01_30"], ["2009_01_30"], ["2009_01_30"]]),
        (
            2,
            2,
            [
                ["2009_01_27", "2009_01_28", "2009_01_29", "2009_01_30", "2009_01_31"],
                ["2009_01_28", "2009_01_29", "2009_01_30", "2009_01_31"],
                ["2009_01_28", "2009_01_29", "2009_01_30", "2009_01_31"],
            ],
        ),
    ],
)
def test_match_gdf_to_img_coll_ids_by_image_count(movebank_relocations_fixed_subset, n_before, n_after, output_list):
    results = ecoscope.io.eetools._match_gdf_to_img_coll_ids(
        gdf=movebank_relocations_fixed_subset,
        time_col="fixtime",
        img_coll=ee.ImageCollection("MODIS/MCD43A4_006_NDVI").select("NDVI"),
        output_col_name="img_ids",
        n_before=n_before,
        n_after=n_after,
        n="images",
    )["img_ids"].to_list()

    # midnight - exact match case
    assert results[0] == output_list[0]
    # am
    assert results[1] == output_list[1]
    # pm
    assert results[2] == output_list[2]


@pytest.mark.parametrize(
    "n_before, n_after, output_list",
    [
        (0, 0, [["2009_01_29"], [], []]),
        (1, 0, [["2009_01_28", "2009_01_29"], ["2009_01_29"], ["2009_01_29"]]),
        (0, 1, [["2009_01_29", "2009_01_30"], ["2009_01_30"], ["2009_01_30"]]),
        (
            2,
            2,
            [
                ["2009_01_27", "2009_01_28", "2009_01_29", "2009_01_30", "2009_01_31"],
                ["2009_01_28", "2009_01_29", "2009_01_30", "2009_01_31"],
                ["2009_01_28", "2009_01_29", "2009_01_30", "2009_01_31"],
            ],
        ),
    ],
)
def test_match_gdf_to_img_coll_ids_by_day(movebank_relocations_fixed_subset, n_before, n_after, output_list):
    output = ecoscope.io.eetools._match_gdf_to_img_coll_ids(
        gdf=movebank_relocations_fixed_subset,
        time_col="fixtime",
        img_coll=ee.ImageCollection("MODIS/MCD43A4_006_NDVI").select("NDVI"),
        output_col_name="img_ids",
        n_before=n_before,
        n_after=n_after,
        n="days",
    )

    print(output[["fixtime", "img_ids"]])
    results = output["img_ids"].to_list()

    # midnight - exact match case
    assert results[0] == output_list[0]
    # am
    assert results[1] == output_list[1]
    # pm
    assert results[2] == output_list[2]
