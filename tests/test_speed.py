import ecoscope
from ecoscope.analysis import speed


def test_speed_geoseries(movebank_relocations):
    trajectory = ecoscope.base.Trajectory.from_relocations(movebank_relocations)
    sdf = speed.SpeedDataFrame.from_trajectory(trajectory)
    assert not sdf.geometry.is_empty.any()
    assert not sdf.geometry.isna().any()
