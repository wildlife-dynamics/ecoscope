import geopandas as gpd
import geopandas.testing
import numpy as np
import pandas as pd
import pandas.testing
import pytest

import ecoscope


def test_redundant_columns_in_trajectory(movbank_relocations):
    # test there is no redundant column in trajectory
    trajectory = ecoscope.base.Trajectory.from_relocations(movbank_relocations)
    assert "extra__fixtime" not in trajectory
    assert "extra___fixtime" not in trajectory
    assert "extra___geometry" not in trajectory


def test_relocs_speedfilter(movbank_relocations):
    relocs_speed_filter = ecoscope.base.RelocsSpeedFilter(max_speed_kmhr=8)
    relocs_after_filter = movbank_relocations.apply_reloc_filter(relocs_speed_filter)
    relocs_after_filter.remove_filtered(inplace=True)
    assert movbank_relocations.shape[0] != relocs_after_filter.shape[0]


def test_relocs_distancefilter(movbank_relocations):
    relocs_speed_filter = ecoscope.base.RelocsDistFilter(min_dist_km=1.0, max_dist_km=6.0)
    relocs_after_filter = movbank_relocations.apply_reloc_filter(relocs_speed_filter)
    relocs_after_filter.remove_filtered(inplace=True)
    assert movbank_relocations.shape[0] != relocs_after_filter.shape[0]


def test_relocations_from_gdf_preserve_fields(movbank_relocations):
    gpd.testing.assert_geodataframe_equal(movbank_relocations, ecoscope.base.Relocations.from_gdf(movbank_relocations))


def test_displacement_property(movbank_relocations):
    trajectory = ecoscope.base.Trajectory.from_relocations(movbank_relocations)
    expected = pd.Series(
        [2633.760505, 147749.545621],
        index=pd.Index(["Habiba", "Salif Keita"], name="groupby_col"),
    )
    pd.testing.assert_series_equal(
        trajectory.groupby("groupby_col").apply(ecoscope.base.Trajectory.get_displacement),
        expected,
    )


def test_tortuosity(movbank_relocations):
    trajectory = ecoscope.base.Trajectory.from_relocations(movbank_relocations)
    expected = pd.Series(
        [51.65388458528601, 75.96149479123005],
        index=pd.Index(["Habiba", "Salif Keita"], name="groupby_col"),
    )
    pd.testing.assert_series_equal(
        trajectory.groupby("groupby_col").apply(ecoscope.base.Trajectory.get_tortuosity),
        expected,
    )


def test_turn_angle(movbank_relocations):
    trajectory = ecoscope.base.Trajectory.from_relocations(movbank_relocations)
    trajectory = trajectory.loc[trajectory.groupby_col == "Habiba"].head(5)
    trajectory["heading"] = [0, 90, 120, 60, 300]
    turn_angle = trajectory.get_turn_angle()

    expected = pd.Series(
        [np.nan, 90, 30, -60, -120],
        dtype=np.float64,
        index=pd.Index([368706890, 368706891, 368706892, 368706893, 368706894], name="event-id"),
        name="turn_angle",
    )
    pandas.testing.assert_series_equal(turn_angle, expected)

    # Test filtering by dropping a row with index: 368706892.
    trajectory.drop(368706892, inplace=True)
    turn_angle = trajectory.get_turn_angle()
    expected = pd.Series(
        [np.nan, 90, np.nan, -120],
        dtype=np.float64,
        index=pd.Index([368706890, 368706891, 368706893, 368706894], name="event-id"),
        name="turn_angle",
    )

    pandas.testing.assert_series_equal(turn_angle, expected)


def test_sampling(movbank_relocations):
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
    movbank_trajectory = ecoscope.base.Trajectory.from_relocations(movbank_relocations)
    noncontiguous_trajectory = pd.concat([movbank_trajectory[0:3], movbank_trajectory[5:8], movbank_trajectory[10:]])

    upsampled_noncontiguous = noncontiguous_trajectory.upsample(60 * 30)
    upsampled_withtolerance = noncontiguous_trajectory.upsample(60 * 30, tolerance=60 * 60 * 3)
    downsampled_relocs_noint = movbank_trajectory.downsample(interval=60 * 60 * 3, tolerance=60 * 15)
    downsampled_relocs_int = movbank_trajectory.downsample(interval=60 * 60 * 3, interpolation=True)

    expected_noncontiguous = gpd.read_feather("tests/test_output/upsampled_noncontiguous.feather")
    expected_withtolerance = gpd.read_feather("tests/test_output/upsampled_withtolerance.feather")
    expected_downsample_noint = gpd.read_feather("tests/test_output/downsampled_relocs_noint.feather")
    expected_downsample_int = gpd.read_feather("tests/test_output/downsampled_relocs.feather")

    gpd.testing.assert_geodataframe_equal(upsampled_noncontiguous, expected_noncontiguous, check_less_precise=True)
    gpd.testing.assert_geodataframe_equal(upsampled_withtolerance, expected_withtolerance, check_less_precise=True)
    gpd.testing.assert_geodataframe_equal(downsampled_relocs_noint, expected_downsample_noint, check_less_precise=True)
    gpd.testing.assert_geodataframe_equal(downsampled_relocs_int, expected_downsample_int, check_less_precise=True)


@pytest.mark.filterwarnings("ignore:Target with index", 'ignore: ERFA function "utctai"')
def test_daynight_ratio(movbank_relocations):
    trajectory = ecoscope.base.Trajectory.from_relocations(movbank_relocations)
    expected = pd.Series(
        [
            2.212816,
            0.638830,
        ],
        index=pd.Index(["Habiba", "Salif Keita"], name="groupby_col"),
    )
    pd.testing.assert_series_equal(
        trajectory.groupby("groupby_col").apply(ecoscope.base.Trajectory.get_daynight_ratio),
        expected,
    )
