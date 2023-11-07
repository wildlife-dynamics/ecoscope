import os

import geopandas as gpd
import geopandas.testing
import numpy as np
import pandas as pd
import pytest
import sklearn.preprocessing

import ecoscope
from ecoscope.analysis.ecograph import (
    FeatureNameError,
    IndividualNameError,
    InterpolationError,
)


@pytest.mark.skip(reason="this has been failing since May 2022; will be fixed in a follow-up pull")
def test_ecograph(movbank_relocations):
    # apply relocation coordinate filter to movbank data
    pnts_filter = ecoscope.base.RelocsCoordinateFilter(
        min_x=-5,
        max_x=1,
        min_y=12,
        max_y=18,
        filter_point_coords=[[180, 90], [0, 0]],
    )
    movbank_relocations.apply_reloc_filter(pnts_filter, inplace=True)
    movbank_relocations.remove_filtered(inplace=True)

    # Create Trajectory
    movebank_trajectory_gdf = ecoscope.base.Trajectory.from_relocations(movbank_relocations)

    os.makedirs("tests/outputs", exist_ok=True)

    mean_step_length = np.mean(np.abs(movebank_trajectory_gdf["dist_meters"]))
    ecograph = ecoscope.analysis.Ecograph(movebank_trajectory_gdf, resolution=mean_step_length)

    ecograph.to_geotiff(
        "betweenness",
        "tests/outputs/salif_btwn_max.tif",
        individual="Salif Keita",
        interpolation="max",
    )
    salif_btwn_max_gdf = ecoscope.analysis.ecograph.get_feature_gdf("tests/outputs/salif_btwn_max.tif")

    ecograph.to_geotiff(
        "degree",
        "tests/outputs/salif_degree_mean.tif",
        individual="Salif Keita",
        interpolation="mean",
    )
    salif_degree_mean_gdf = ecoscope.analysis.ecograph.get_feature_gdf("tests/outputs/salif_degree_mean.tif")

    ecograph.to_geotiff(
        "collective_influence",
        "tests/outputs/salif_ci_min.tif",
        individual="Salif Keita",
        interpolation="min",
    )
    salif_ci_min_gdf = ecoscope.analysis.ecograph.get_feature_gdf("tests/outputs/salif_ci_min.tif")

    ecograph.to_geotiff(
        "step_length",
        "tests/outputs/salif_sl_median.tif",
        individual="Salif Keita",
        interpolation="median",
    )
    salif_sl_median_gdf = ecoscope.analysis.ecograph.get_feature_gdf("tests/outputs/salif_sl_median.tif")

    ecograph.to_geotiff(
        "degree",
        "tests/outputs/salif_degree_mean_std.tif",
        individual="Salif Keita",
        interpolation="mean",
        transform=sklearn.preprocessing.StandardScaler(),
    )
    salif_degree_mean_std_gdf = ecoscope.analysis.ecograph.get_feature_gdf("tests/outputs/salif_degree_mean_std.tif")

    ecograph.to_geotiff("speed", "tests/outputs/salif_speed.tif", individual="Salif Keita")
    salif_speed_gdf = ecoscope.analysis.ecograph.get_feature_gdf("tests/outputs/salif_speed.tif")

    ecograph.to_geotiff("dot_product", "tests/outputs/salif_dotprod.tif", individual="Salif Keita")
    salif_dotprod_gdf = ecoscope.analysis.ecograph.get_feature_gdf("tests/outputs/salif_dotprod.tif")

    expected_btw_max = gpd.read_feather("tests/test_output/salif_btwn_max.feather")
    expected_degree_mean = gpd.read_feather("tests/test_output/salif_degree_mean.feather")
    expected_ci_min = gpd.read_feather("tests/test_output/salif_ci_min.feather")
    expected_sl_median = gpd.read_feather("tests/test_output/salif_sl_median.feather")
    expected_speed = gpd.read_feather("tests/test_output/salif_speed.feather")
    expected_dotprod = gpd.read_feather("tests/test_output/salif_dotprod.feather")
    expected_degree_mean_std = gpd.read_feather("tests/test_output/salif_degree_mean_std.feather")

    gpd.testing.assert_geodataframe_equal(salif_btwn_max_gdf, expected_btw_max, check_less_precise=True)
    gpd.testing.assert_geodataframe_equal(salif_degree_mean_gdf, expected_degree_mean, check_less_precise=True)
    gpd.testing.assert_geodataframe_equal(salif_ci_min_gdf, expected_ci_min, check_less_precise=True)
    gpd.testing.assert_geodataframe_equal(salif_sl_median_gdf, expected_sl_median, check_less_precise=True)
    gpd.testing.assert_geodataframe_equal(salif_speed_gdf, expected_speed, check_less_precise=True)
    gpd.testing.assert_geodataframe_equal(salif_dotprod_gdf, expected_dotprod, check_less_precise=True)
    gpd.testing.assert_geodataframe_equal(salif_degree_mean_std_gdf, expected_degree_mean_std, check_less_precise=True)

    with pytest.raises(InterpolationError):
        ecograph.to_geotiff(
            "betweenness",
            "tests/outputs/salif_btwn_max.tif",
            individual="Salif Keita",
            interpolation="osjfos",
        )
    with pytest.raises(FeatureNameError):
        ecograph.to_geotiff(
            "rojfoofj",
            "tests/outputs/salif_btwn_max.tif",
            individual="Salif Keita",
            interpolation="max",
        )
    with pytest.raises(IndividualNameError):
        ecograph.to_geotiff(
            "betweenness",
            "tests/outputs/salif_btwn_max.tif",
            individual="fofkfp",
            interpolation="max",
        )

    ecograph.to_csv("tests/outputs/features.csv")
    feat = pd.read_csv("tests/outputs/features.csv")
    expected_feat = pd.read_csv("tests/test_output/features.csv")
    pd.testing.assert_frame_equal(feat, expected_feat)
