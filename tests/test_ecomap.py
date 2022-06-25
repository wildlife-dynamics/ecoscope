import ecoscope


def test_ecomap_base():
    ecoscope.mapping.EcoMap()


def test_add_local_geotiff():
    m = ecoscope.mapping.EcoMap()
    m.add_local_geotiff("tests/sample_data/raster/uint8.tif", cmap=None)
    m.add_local_geotiff("tests/sample_data/raster/uint8.tif", cmap="jet")
