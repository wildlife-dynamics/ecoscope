import pytest
from unittest.mock import patch
from ecoscope.io.vector import export_pmtiles


@pytest.mark.parametrize(
    "input_args, result",
    [
        (
            [],
            [
                "/usr/bin/tippecanoe",
                "-zg",
                "--drop-densest-as-needed",
                "--extend-zooms-if-still-dropping",
                "-l",
                "test",
                "-o",
                "testpath.pmtiles",
                "temp.fgb",
            ],
        ),
        (["arg1", "arg2"], ["/usr/bin/tippecanoe", "arg1", "arg2", "-l", "test", "-o", "testpath.pmtiles", "temp.fgb"]),
    ],
)
@patch("ecoscope.io.vector.which", return_value="/usr/bin/tippecanoe")
@patch("ecoscope.io.vector.Popen")
def test_export_pmtiles(mocked_popen, mocked_which, input_args, result, movbank_relocations):

    export_pmtiles(gdf=movbank_relocations, filepath="testpath.pmtiles", layer_name="test", args=input_args)

    args, kwargs = mocked_popen.call_args

    assert args[0] == result


@pytest.mark.parametrize(
    "input_args, result",
    [
        (
            [],
            [
                "/usr/bin/ogr2ogr",
                "-lco",
                "NAME=test",
                "-progress",
                "-f",
                "PMTiles",
                "-dsco",
                "MAXZOOM=20",
                "testpath.pmtiles",
                "temp.fgb",
            ],
        ),
        (["arg1", "arg2"], ["/usr/bin/ogr2ogr", "-lco", "NAME=test", "arg1", "arg2", "testpath.pmtiles", "temp.fgb"]),
    ],
)
@patch("ecoscope.io.vector.which", return_value="/usr/bin/ogr2ogr")
@patch("ecoscope.io.vector.Popen")
def test_export_pmtiles_gdal(mocked_popen, mocked_which, input_args, result, movbank_relocations):

    export_pmtiles(
        gdf=movbank_relocations, filepath="testpath.pmtiles", layer_name="test", use_gdal=True, args=input_args
    )

    args, kwargs = mocked_popen.call_args

    assert args[0] == result


@patch("ecoscope.io.vector.which", return_value=None)
def test_export_pmtiles_exception(mocked_which, movbank_relocations):

    with pytest.raises(FileNotFoundError):
        export_pmtiles(gdf=movbank_relocations, filepath="testpath.pmtiles", layer_name="test")
