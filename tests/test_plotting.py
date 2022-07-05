import ecoscope


def test_ecoplot(movbank_relocations):
    traj = ecoscope.base.Trajectory.from_relocations(movbank_relocations)
    grouped1 = traj[:1000].groupby("groupby_col")
    grouped2 = traj[2000:3000].groupby("groupby_col")

    epdata1 = ecoscope.plotting.EcoPlotData(grouped1, "segment_start", "speed_kmhr", line=dict(color="blue"))
    epdata2 = ecoscope.plotting.EcoPlotData(grouped2, "segment_start", "speed_kmhr", line=dict(color="blue"))
    epdata3 = ecoscope.plotting.EcoPlotData(grouped2, "segment_start", "timespan_seconds", line=dict(color="red"))

    ecoscope.plotting.ecoplot([epdata1, epdata2, epdata3], "My Plot")

def test_mcp(movbank_relocations):
    ecoscope.plotting.mcp(movbank_relocations)

def test_speedplot(movbank_relocations):
    traj = ecoscope.base.Trajectory.from_relocations(movbank_relocations)
    ecoscope.plotting.speed(traj)