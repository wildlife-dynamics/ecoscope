import geopandas as gpd
import numpy as np
import numpy.testing

import ecoscope


def test_is_night(movbank_relocations):
    is_night = ecoscope.analysis.astronomy.is_night(movbank_relocations.geometry, movbank_relocations.fixtime)
    expected_isnight = np.array(gpd.read_feather("tests/test_output/movbank_isnight.feather")["is_night"])
    np.testing.assert_array_equal(is_night, expected_isnight)
