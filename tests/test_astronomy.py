import geopandas as gpd
import geopandas.testing
import ecoscope

def test_is_night(movbank_relocations):
    movbank_relocations["is_night"] = ecoscope.analysis.astronomy.is_night(movbank_relocations.geometry, movbank_relocations.fixtime)
    expected_isnight = gpd.read_feather("tests/test_output/movbank_isnight.feather")
    gpd.testing.assert_geodataframe_equal(movbank_relocations, expected_isnight)
    