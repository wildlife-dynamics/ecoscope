import geopandas as gpd
import geopandas.testing
import numpy as np
import pandas as pd
import pytest
import sklearn.preprocessing

import ecoscope
from ecoscope.analysis.ecograph import Ecograph, get_feature_gdf


@pytest.fixture
def movebank_trajectory_gdf(movebank_relocations):
    # apply relocation coordinate filter to movebank data
    pnts_filter = ecoscope.base.RelocsCoordinateFilter(
        min_x=-5,
        max_x=1,
        min_y=12,
        max_y=18,
        filter_point_coords=[[180, 90], [0, 0]],
    )
    movebank_relocations.apply_reloc_filter(pnts_filter, inplace=True)
    movebank_relocations.remove_filtered(inplace=True)

    # Create Trajectory
    return ecoscope.base.Trajectory.from_relocations(movebank_relocations)


@pytest.fixture
def movebank_ecograph(movebank_trajectory_gdf):
    mean_step_length = np.mean(np.abs(movebank_trajectory_gdf["dist_meters"]))
    return Ecograph(movebank_trajectory_gdf, resolution=mean_step_length)


@pytest.mark.parametrize(
    "feature, interpolation, transform, output_file, validation_file",
    [
        ("betweenness", "max", None, "salif_btwn_max.tif", "salif_btwn_max.feather"),
        ("degree", "mean", None, "salif_degree_mean.tif", "salif_degree_mean.feather"),
        ("collective_influence", "min", None, "salif_ci_min.tif", "salif_ci_min.feather"),
        ("step_length", "median", None, "salif_sl_median.tif", "salif_sl_median.feather"),
        (
            "degree",
            "mean",
            sklearn.preprocessing.StandardScaler(),
            "salif_degree_mean_std.tif",
            "salif_degree_mean_std.feather",
        ),
        ("speed", None, None, "salif_speed.tif", "salif_speed.feather"),
        ("dot_product", None, None, "salif_dotprod.tif", "salif_dotprod.feather"),
    ],
)
def test_ecograph_to_geotiff(movebank_ecograph, feature, interpolation, transform, output_file, validation_file):
    movebank_ecograph.to_geotiff(
        feature,
        f"tests/outputs/{output_file}",
        individual="Salif Keita",
        interpolation=interpolation,
        transform=transform,
    )
    gdf_from_tiff = get_feature_gdf(f"tests/outputs/{output_file}")

    # expected_gdf = gpd.read_feather(f"tests/test_output/{validation_File}")
    expected_gdf = gpd.read_feather(f"tests/reference_data/{validation_file}")
    gpd.testing.assert_geodataframe_equal(gdf_from_tiff, expected_gdf)


def test_ecograph_to_geotiff_with_error(movebank_ecograph):
    with pytest.raises(NotImplementedError):
        movebank_ecograph.to_geotiff(
            "betweenness",
            "tests/outputs/salif_btwn_max.tif",
            individual="Salif Keita",
            interpolation="osjfos",
        )

    with pytest.raises(ValueError):
        movebank_ecograph.to_geotiff(
            "rojfoofj",
            "tests/outputs/salif_btwn_max.tif",
            individual="Salif Keita",
            interpolation="max",
        )
        movebank_ecograph.to_geotiff(
            "betweenness",
            "tests/outputs/salif_btwn_max.tif",
            individual="fofkfp",
            interpolation="max",
        )


def test_ecograph_to_csv(movebank_ecograph):
    movebank_ecograph.to_csv("tests/outputs/features.csv")
    feat = pd.read_csv("tests/outputs/features.csv")
    # expected_feat = pd.read_csv("tests/test_output/features.csv")

    for item in ["individual_name", "grid_id"] + movebank_ecograph.features:
        assert item in feat.columns

    assert len(movebank_ecograph.graphs["Salif Keita"]) == len(feat)
