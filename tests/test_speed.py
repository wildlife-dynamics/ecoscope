import geopandas as gpd
import geopandas.testing
import numpy as np
import pandas as pd
import pandas.testing
import pytest
import ecoscope
from ecoscope.analysis import speed


def test_speed_geoseries(movbank_relocations):
    trajectory = ecoscope.base.Trajectory.from_relocations(movbank_relocations)
    sdf = speed.SpeedDataFrame.from_trajectory(trajectory)
    print(sdf.geometry.is_empty)
    print(sdf.geometry.isna())
    assert all(~sdf.geometry.is_empty)
    assert all(~sdf.geometry.isna())
