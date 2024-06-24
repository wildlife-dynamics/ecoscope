import geopandas as gpd
import geopandas.testing
import numpy as np
import pandas as pd
import pandas.testing
import pytest

import ecoscope


@pytest.mark.skipif(not pytest.earthranger, reason="No connection to EarthRanger")
def test_trajectory_is_not_empty(er_io):
    # test there is actually data in trajectory
    relocations = er_io.get_subjectgroup_observations(subject_group_name=er_io.GROUP_NAME)
    trajectory = ecoscope.base.Trajectory.from_relocations(relocations)
    assert not trajectory.empty


@pytest.mark.skipif(not pytest.earthranger, reason="No connection to EarthRanger")
def test_redundant_columns_in_trajectory(er_io):
    # test there is no redundant column in trajectory
    relocations = er_io.get_subjectgroup_observations(subject_group_name=er_io.GROUP_NAME)
    trajectory = ecoscope.base.Trajectory.from_relocations(relocations)
    assert "extra__fixtime" not in trajectory
    assert "extra___fixtime" not in trajectory
    assert "extra___geometry" not in trajectory


@pytest.mark.skipif(not pytest.earthranger, reason="No connection to EarthRanger")
def test_relocs_speedfilter(er_io):
    relocations = er_io.get_subjectgroup_observations(subject_group_name=er_io.GROUP_NAME)
    relocs_speed_filter = ecoscope.base.RelocsSpeedFilter(max_speed_kmhr=8)
    relocs_after_filter = relocations.apply_reloc_filter(relocs_speed_filter)
    relocs_after_filter.remove_filtered(inplace=True)
    assert relocations.shape[0] != relocs_after_filter.shape[0]


@pytest.mark.skipif(not pytest.earthranger, reason="No connection to EarthRanger")
def test_relocs_distancefilter(er_io):
    relocations = er_io.get_subjectgroup_observations(subject_group_name=er_io.GROUP_NAME)
    relocs_speed_filter = ecoscope.base.RelocsDistFilter(min_dist_km=1.0, max_dist_km=6.0)
    relocs_after_filter = relocations.apply_reloc_filter(relocs_speed_filter)
    relocs_after_filter.remove_filtered(inplace=True)
    assert relocations.shape[0] != relocs_after_filter.shape[0]


@pytest.mark.skipif(not pytest.earthranger, reason="No connection to EarthRanger")
def test_relocations_from_gdf_preserve_fields(er_io):
    relocations = er_io.get_subjectgroup_observations(subject_group_name=er_io.GROUP_NAME)
    gpd.testing.assert_geodataframe_equal(relocations, ecoscope.base.Relocations.from_gdf(relocations))


def test_displacement_property(movebank_relocations):
    trajectory = ecoscope.base.Trajectory.from_relocations(movebank_relocations)
    expected = pd.Series(
        [2633.760505, 147749.545621],
        index=pd.Index(["Habiba", "Salif Keita"], name="groupby_col"),
    )
    pd.testing.assert_series_equal(
        trajectory.groupby("groupby_col")[trajectory.columns].apply(ecoscope.base.Trajectory.get_displacement),
        expected,
    )


def test_tortuosity(movebank_relocations):
    trajectory = ecoscope.base.Trajectory.from_relocations(movebank_relocations)
    expected = pd.Series(
        [51.65388458528601, 75.96149479123005],
        index=pd.Index(["Habiba", "Salif Keita"], name="groupby_col"),
    )
    pd.testing.assert_series_equal(
        trajectory.groupby("groupby_col")[trajectory.columns].apply(
            ecoscope.base.Trajectory.get_tortuosity, include_groups=False
        ),
        expected,
    )


def test_turn_angle(movebank_relocations):
    trajectory = ecoscope.base.Trajectory.from_relocations(movebank_relocations)
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


def test_sampling(movebank_relocations):
    relocs_1 = ecoscope.base.Relocations.from_gdf(
        gpd.GeoDataFrame(
            {"fixtime": pd.date_range(0, periods=1000, freq="1s", tz="utc")},
            geometry=gpd.points_from_xy(x=np.zeros(1000), y=np.linspace(0, 1, 1000)),
            crs=4326,
        )
    )
    traj_1 = ecoscope.base.Trajectory.from_relocations(relocs_1).loc[::2]
    upsampled_noncontiguous_1 = traj_1.upsample("3600S")

    relocs_2 = ecoscope.base.Relocations.from_gdf(
        gpd.GeoDataFrame(
            {"fixtime": pd.date_range(0, periods=10000, freq="2s", tz="utc")},
            geometry=gpd.points_from_xy(x=np.zeros(10000), y=np.linspace(0, 1, 10000)),
            crs=4326,
        )
    )
    traj_2 = ecoscope.base.Trajectory.from_relocations(relocs_2).loc[::2]
    upsampled_noncontiguous_2 = traj_2.upsample("3S")

    pnts_filter = ecoscope.base.RelocsCoordinateFilter(
        min_x=-5,
        max_x=1,
        min_y=12,
        max_y=18,
        filter_point_coords=[[180, 90], [0, 0]],
    )
    movebank_relocations.apply_reloc_filter(pnts_filter, inplace=True)
    movebank_relocations.remove_filtered(inplace=True)
    trajectory = ecoscope.base.Trajectory.from_relocations(movebank_relocations)
    downsampled_relocs_noint = trajectory.downsample("10800S", tolerance="900S")
    downsampled_relocs_int = trajectory.downsample("10800S", interpolation=True)

    expected_noncontiguous_1 = gpd.read_feather("tests/test_output/upsampled_noncontiguous_1.feather")
    expected_noncontiguous_2 = gpd.read_feather("tests/test_output/upsampled_noncontiguous_2.feather")
    expected_downsample_noint = gpd.read_feather("tests/test_output/downsampled_relocs_noint.feather")
    expected_downsample_int = gpd.read_feather("tests/test_output/downsampled_relocs.feather")

    gpd.testing.assert_geodataframe_equal(upsampled_noncontiguous_1, expected_noncontiguous_1, check_less_precise=True)
    gpd.testing.assert_geodataframe_equal(upsampled_noncontiguous_2, expected_noncontiguous_2, check_less_precise=True)
    gpd.testing.assert_geodataframe_equal(downsampled_relocs_noint, expected_downsample_noint, check_less_precise=True)
    gpd.testing.assert_geodataframe_equal(downsampled_relocs_int, expected_downsample_int, check_less_precise=True)


def test_edf_filter(movebank_relocations):
    movebank_relocations["junk_status"] = True

    empty = movebank_relocations.remove_filtered()
    assert len(empty) == 0

    reset = movebank_relocations.reset_filter().remove_filtered()
    assert len(reset) > 0


def test_relocs_from_gdf_with_warnings():
    df = pd.read_feather("tests/sample_data/vector/movebank_data.feather")
    geometry = gpd.points_from_xy(df.pop("location-long"), df.pop("location-lat"))

    gdf = gpd.GeoDataFrame(df, geometry=geometry)

    with pytest.warns(UserWarning, match="CRS was not set"):
        ecoscope.base.Relocations.from_gdf(gdf, time_col="timestamp")

    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs=4326)
    gdf["timestamp"] = pd.to_datetime(gdf["timestamp"])

    with pytest.warns(UserWarning, match="timestamp is not timezone aware"):
        ecoscope.base.Relocations.from_gdf(gdf, time_col="timestamp")

    gdf["timestamp"] = "1/1/2000"

    with pytest.warns(UserWarning, match="timestamp is not of type datetime64"):
        ecoscope.base.Relocations.from_gdf(gdf, time_col="timestamp")


def test_apply_traj_filter(movebank_relocations):
    trajectory = ecoscope.base.Trajectory.from_relocations(movebank_relocations)

    min_length = 0.2
    max_length = 6000
    min_time = 100
    max_time = 300000
    min_speed = 0.1
    max_speed = 5

    traj_seg_filter = ecoscope.base.TrajSegFilter(
        min_length_meters=min_length,
        max_length_meters=max_length,
        min_time_secs=min_time,
        max_time_secs=max_time,
        min_speed_kmhr=min_speed,
        max_speed_kmhr=max_speed,
    )

    filtered = trajectory.apply_traj_filter(traj_seg_filter)
    filtered.remove_filtered(inplace=True)

    assert filtered["dist_meters"].min() >= min_length
    assert filtered["dist_meters"].max() <= max_length

    assert filtered["timespan_seconds"].min() >= min_time
    assert filtered["timespan_seconds"].max() <= max_time

    assert filtered["speed_kmhr"].min() >= min_speed
    assert filtered["speed_kmhr"].max() <= max_speed
