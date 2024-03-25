import os
from shutil import which
from subprocess import Popen, PIPE


def export_pmtiles(gdf, filepath, layer_name="layer1", use_gdal=False, args=[]):
    """
    Exports a given gdf as a pmtiles archive using the local install of either tippecanoe or gdal
    Will search for tippecanoe on path and use that, falling back to gdal if it's not found
    This works by first dumping the given gdf as a FlatGeobuf to provide as input to the chosen tool

    If args is not provided, the following defaults are used:
        tippecanoe -zg --drop-densest-as-needed --extend-zooms-if-still-dropping -l <layer_name> -o <filepath> temp.fgb
        ogr2ogr -progress -f  PMTiles -dsco MAXZOOM=20 -lco NAME=<layer_name> <filepath> temp.fgb

    Parameters
    ----------
    gdf : GeoDataFrame
        The gdf to export
    filepath : str
        The output filepath, if using tippecanoe this must end with .pmtiles
    layer_name : str, optional
        The layer name of the feature collection within the exported PMTiles, default is layer1
    use_gdal : bool, optional
        Force the use of gdal, default is False
    args : list, optional
        Override the default command line args used, this will not override the provided filepath or layer_name
    """

    tempfile = "temp.fgb"
    gdf.to_file(tempfile, "FlatGeobuf")

    default_args_tippecanoe = ["-zg", "--drop-densest-as-needed", "--extend-zooms-if-still-dropping"]
    default_args_ogr = ["-progress", "-f", "PMTiles", "-dsco", "MAXZOOM=20"]

    if (cmd := which("tippecanoe")) and not use_gdal:
        if len(args) == 0:
            args = default_args_tippecanoe

        if layer_name != "":
            args.extend(["-l", layer_name])

        args.extend(["-o", filepath, tempfile])

    elif cmd := which("ogr2ogr"):
        if len(args) == 0:
            args = default_args_ogr

        if layer_name != "":
            args = ["-lco", "NAME=" + layer_name] + args

        args.extend([filepath, tempfile])

    else:
        os.remove(tempfile)
        raise FileNotFoundError("no tippecanoe or ogr2ogr was found on system path")

    args = [cmd] + args

    with Popen(args, stdout=PIPE, stderr=PIPE) as proc:
        while proc.poll() is None:
            print(proc.stdout.read1().decode("utf-8"), end="", flush=True)
            print(proc.stderr.read1().decode("utf-8"), end="", flush=True)

    os.remove(tempfile)

    return filepath
