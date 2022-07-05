import geopandas as gpd
import geopandas.testing
import ecoscope

def test_speed_dataframe(movbank_relocations):
    traj = ecoscope.base.Trajectory.from_relocations(movbank_relocations)
    sdf = ecoscope.analysis.speed.SpeedDataFrame.from_trajectory(traj)
    sdf_bins = ecoscope.analysis.speed.SpeedDataFrame.from_trajectory(traj, bins=[1, 4, 5, 100])

    expected_sdf = gpd.read_feather("tests/test_output/traj_sdf.feather")
    expected_sdf_bins = gpd.read_feather("tests/test_output/traj_sdf_bins.feather")

    gpd.testing.assert_geodataframe_equal(sdf, expected_sdf)
    gpd.testing.assert_geodataframe_equal(sdf_bins, expected_sdf_bins)