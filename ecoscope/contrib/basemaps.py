"""
MIT License

Copyright (c) 2021, Qiusheng Wu

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

__version__ = "be6ccdfa5450b775724547b6ef50f213c820be85"

"""Module for basemaps. 

More WMS basemaps can be found at the following websites:

1. USGS National Map: https://viewer.nationalmap.gov/services

2. MRLC NLCD Land Cover data: https://www.mrlc.gov/data-services-page

3. FWS NWI Wetlands data: https://www.fws.gov/wetlands/Data/Web-Map-Services.html

"""
import collections
import folium
import xyzservices.providers as xyz

# from box import Box

# Custom XYZ tile services.
xyz_tiles = {
    "OpenStreetMap": {
        "url": "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
        "attribution": "OpenStreetMap",
        "name": "OpenStreetMap",
    },
    "ROADMAP": {
        "url": "https://mt1.google.com/vt/lyrs=m&x={x}&y={y}&z={z}",
        "attribution": "Google",
        "name": "Google Maps",
    },
    "SATELLITE": {
        "url": "https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
        "attribution": "Google",
        "name": "Google Satellite",
    },
    "TERRAIN": {
        "url": "https://mt1.google.com/vt/lyrs=p&x={x}&y={y}&z={z}",
        "attribution": "Google",
        "name": "Google Terrain",
    },
    "HYBRID": {
        "url": "https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}",
        "attribution": "Google",
        "name": "Google Satellite",
    },
}

# Custom WMS tile services.
wms_tiles = {
    "FWS NWI Wetlands": {
        "url": "https://www.fws.gov/wetlands/arcgis/services/Wetlands/MapServer/WMSServer?",
        "layers": "1",
        "name": "FWS NWI Wetlands",
        "attribution": "FWS",
        "format": "image/png",
        "transparent": True,
    },
    "FWS NWI Wetlands Raster": {
        "url": "https://www.fws.gov/wetlands/arcgis/services/Wetlands_Raster/ImageServer/WMSServer?",
        "layers": "0",
        "name": "FWS NWI Wetlands Raster",
        "attribution": "FWS",
        "format": "image/png",
        "transparent": True,
    },
    "NLCD 2019 CONUS Land Cover": {
        "url": "https://www.mrlc.gov/geoserver/mrlc_display/NLCD_2019_Land_Cover_L48/wms?",
        "layers": "NLCD_2019_Land_Cover_L48",
        "name": "NLCD 2019 CONUS Land Cover",
        "attribution": "MRLC",
        "format": "image/png",
        "transparent": True,
    },
    "NLCD 2016 CONUS Land Cover": {
        "url": "https://www.mrlc.gov/geoserver/mrlc_display/NLCD_2016_Land_Cover_L48/wms?",
        "layers": "NLCD_2016_Land_Cover_L48",
        "name": "NLCD 2016 CONUS Land Cover",
        "attribution": "MRLC",
        "format": "image/png",
        "transparent": True,
    },
    "NLCD 2013 CONUS Land Cover": {
        "url": "https://www.mrlc.gov/geoserver/mrlc_display/NLCD_2013_Land_Cover_L48/wms?",
        "layers": "NLCD_2013_Land_Cover_L48",
        "name": "NLCD 2013 CONUS Land Cover",
        "attribution": "MRLC",
        "format": "image/png",
        "transparent": True,
    },
    "NLCD 2011 CONUS Land Cover": {
        "url": "https://www.mrlc.gov/geoserver/mrlc_display/NLCD_2011_Land_Cover_L48/wms?",
        "layers": "NLCD_2011_Land_Cover_L48",
        "name": "NLCD 2011 CONUS Land Cover",
        "attribution": "MRLC",
        "format": "image/png",
        "transparent": True,
    },
    "NLCD 2008 CONUS Land Cover": {
        "url": "https://www.mrlc.gov/geoserver/mrlc_display/NLCD_2008_Land_Cover_L48/wms?",
        "layers": "NLCD_2008_Land_Cover_L48",
        "name": "NLCD 2008 CONUS Land Cover",
        "attribution": "MRLC",
        "format": "image/png",
        "transparent": True,
    },
    "NLCD 2006 CONUS Land Cover": {
        "url": "https://www.mrlc.gov/geoserver/mrlc_display/NLCD_2006_Land_Cover_L48/wms?",
        "layers": "NLCD_2006_Land_Cover_L48",
        "name": "NLCD 2006 CONUS Land Cover",
        "attribution": "MRLC",
        "format": "image/png",
        "transparent": True,
    },
    "NLCD 2004 CONUS Land Cover": {
        "url": "https://www.mrlc.gov/geoserver/mrlc_display/NLCD_2004_Land_Cover_L48/wms?",
        "layers": "NLCD_2004_Land_Cover_L48",
        "name": "NLCD 2004 CONUS Land Cover",
        "attribution": "MRLC",
        "format": "image/png",
        "transparent": True,
    },
    "NLCD 2001 CONUS Land Cover": {
        "url": "https://www.mrlc.gov/geoserver/mrlc_display/NLCD_2001_Land_Cover_L48/wms?",
        "layers": "NLCD_2001_Land_Cover_L48",
        "name": "NLCD 2001 CONUS Land Cover",
        "attribution": "MRLC",
        "format": "image/png",
        "transparent": True,
    },
    "USGS NAIP Imagery": {
        "url": "https://imagery.nationalmap.gov/arcgis/services/USGSNAIPImagery/ImageServer/WMSServer?",
        "layers": "USGSNAIPImagery:NaturalColor",
        "name": "USGS NAIP Imagery",
        "attribution": "USGS",
        "format": "image/png",
        "transparent": True,
    },
    "USGS NAIP Imagery False Color": {
        "url": "https://imagery.nationalmap.gov/arcgis/services/USGSNAIPImagery/ImageServer/WMSServer?",
        "layers": "USGSNAIPImagery:FalseColorComposite",
        "name": "USGS NAIP Imagery False Color",
        "attribution": "USGS",
        "format": "image/png",
        "transparent": True,
    },
    "USGS NAIP Imagery NDVI": {
        "url": "https://imagery.nationalmap.gov/arcgis/services/USGSNAIPImagery/ImageServer/WMSServer?",
        "layers": "USGSNAIPImagery:NDVI_Color",
        "name": "USGS NAIP Imagery NDVI",
        "attribution": "USGS",
        "format": "image/png",
        "transparent": True,
    },
    "USGS Hydrography": {
        "url": "https://basemap.nationalmap.gov/arcgis/services/USGSHydroCached/MapServer/WMSServer?",
        "layers": "0",
        "name": "USGS Hydrography",
        "attribution": "USGS",
        "format": "image/png",
        "transparent": True,
    },
    "USGS 3DEP Elevation": {
        "url": "https://elevation.nationalmap.gov/arcgis/services/3DEPElevation/ImageServer/WMSServer?",
        "layers": "33DEPElevation:Hillshade Elevation Tinted",
        "name": "USGS 3DEP Elevation",
        "attribution": "USGS",
        "format": "image/png",
        "transparent": True,
    },
    "ESA WorldCover 2020": {
        "url": "https://services.terrascope.be/wms/v2",
        "layers": "WORLDCOVER_2020_MAP",
        "name": "ESA Worldcover 2020",
        "attribution": "ESA",
        "format": "image/png",
        "transparent": True,
    },
    "ESA WorldCover 2020 S2 FCC": {
        "url": "https://services.terrascope.be/wms/v2",
        "layers": "WORLDCOVER_2020_S2_FCC",
        "name": "ESA Worldcover 2020 S2 FCC",
        "attribution": "ESA",
        "format": "image/png",
        "transparent": True,
    },
    "ESA WorldCover 2020 S2 TCC": {
        "url": "https://services.terrascope.be/wms/v2",
        "layers": "WORLDCOVER_2020_S2_TCC",
        "name": "ESA Worldcover 2020 S2 TCC",
        "attribution": "ESA",
        "format": "image/png",
        "transparent": True,
    },
}


def _unpack_sub_parameters(var, param):
    temp = var
    for sub_param in param.split("."):
        temp = getattr(temp, sub_param)
    return temp


def get_xyz_dict(free_only=True):
    """Returns a dictionary of xyz services.

    Args:
        free_only (bool, optional): Whether to return only free xyz tile services that do not require an access token. Defaults to True.

    Returns:
        dict: A dictionary of xyz services.
    """

    xyz_dict = {}
    for item in xyz.values():
        try:
            name = item["name"]
            tile = _unpack_sub_parameters(xyz, name)
            if _unpack_sub_parameters(xyz, name).requires_token():
                if free_only:
                    pass
                else:
                    xyz_dict[name] = tile
            else:
                xyz_dict[name] = tile

        except Exception:
            for sub_item in item:
                name = item[sub_item]["name"]
                tile = _unpack_sub_parameters(xyz, name)
                if _unpack_sub_parameters(xyz, name).requires_token():
                    if free_only:
                        pass
                    else:
                        xyz_dict[name] = tile
                else:
                    xyz_dict[name] = tile

    xyz_dict = collections.OrderedDict(sorted(xyz_dict.items()))
    return xyz_dict


def xyz_to_folium():
    """Convert xyz tile services to folium tile layers.

    Returns:
        dict: A dictionary of folium tile layers.
    """
    folium_dict = {}

    for key in xyz_tiles:
        name = xyz_tiles[key]["name"]
        url = xyz_tiles[key]["url"]
        attribution = xyz_tiles[key]["attribution"]
        folium_dict[key] = folium.TileLayer(
            tiles=url,
            attr=attribution,
            name=name,
            overlay=True,
            control=True,
            max_zoom=22,
        )

    for key in wms_tiles:
        name = wms_tiles[key]["name"]
        url = wms_tiles[key]["url"]
        layers = wms_tiles[key]["layers"]
        fmt = wms_tiles[key]["format"]
        transparent = wms_tiles[key]["transparent"]
        attribution = wms_tiles[key]["attribution"]
        folium_dict[key] = folium.WmsTileLayer(
            url=url,
            layers=layers,
            name=name,
            attr=attribution,
            fmt=fmt,
            transparent=transparent,
            overlay=True,
            control=True,
        )

    xyz_dict = get_xyz_dict()
    for item in xyz_dict:
        name = xyz_dict[item].name
        url = xyz_dict[item].build_url()
        attribution = xyz_dict[item].attribution
        if "max_zoom" in xyz_dict[item].keys():
            max_zoom = xyz_dict[item]["max_zoom"]
        else:
            max_zoom = 22
        folium_dict[name] = folium.TileLayer(
            tiles=url,
            attr=attribution,
            name=name,
            max_zoom=max_zoom,
            overlay=True,
            control=True,
        )

    return folium_dict
