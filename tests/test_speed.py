import ecoscope


def test_speed_geoseries(movbank_relocations):
    trajectory = ecoscope.base.Trajectory.from_relocations(movbank_relocations)
    sdf = ecoscope.analysis.speed.SpeedDataFrame.from_trajectory(trajectory)
    assert not sdf.geometry.is_empty.any()
    assert not sdf.geometry.isna().any()
