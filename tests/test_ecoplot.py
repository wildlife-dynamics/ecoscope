import numpy as np
from ecoscope.plotting.plot import EcoPlotData, ecoplot, mcp, nsd, speed
from ecoscope.base import Trajectory


def test_ecoplot(movebank_relocations):
    traj = Trajectory.from_relocations(movebank_relocations)
    epd = EcoPlotData(traj.groupby("groupby_col"), "segment_start", "speed_kmhr", line=dict(color="blue"))
    figure = ecoplot([epd], "EcoPlot")

    habiba = traj.loc[traj["groupby_col"] == "Habiba"]
    salif = traj.loc[traj["groupby_col"] == "Salif Keita"]

    assert len(figure.data) == 2

    assert figure.data[0].name == "Habiba"
    assert np.equal(figure.data[0].x, habiba["segment_start"].array).all()
    assert np.equal(figure.data[0].y, habiba["speed_kmhr"].array).all()

    assert figure.data[1].name == "Salif Keita"
    assert np.equal(figure.data[1].x, salif["segment_start"].array).all()
    assert np.equal(figure.data[1].y, salif["speed_kmhr"].array).all()


def test_mcp(movebank_relocations):
    figure = mcp(movebank_relocations)

    assert len(figure.data) == 1
    assert movebank_relocations["fixtime"].iat[0] == figure.data[0].x[0]
    assert movebank_relocations["fixtime"].iat[-1] == figure.data[0].x[-1]


def test_nsd(movebank_relocations):
    figure = nsd(movebank_relocations)

    assert len(figure.data) == 1
    assert len(figure.data[0].x) == len(movebank_relocations)
    assert len(figure.data[0].y) == len(movebank_relocations)


def test_speed(movebank_relocations):
    traj = Trajectory.from_relocations(movebank_relocations)
    figure = speed(traj)

    assert len(figure.data) == 1
    len(figure.data[0].x) == len(traj) * 4
    len(figure.data[0].y) == len(traj) * 4
