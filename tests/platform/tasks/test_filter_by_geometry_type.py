import geopandas as gpd  # type: ignore[import-untyped]
from shapely.geometry import LineString, MultiPolygon, Point, Polygon

from ecoscope.platform.tasks.transformation import filter_by_geometry_type


def _mixed_gdf() -> gpd.GeoDataFrame:
    poly = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    multipoly = MultiPolygon([poly, Polygon([(2, 2), (3, 2), (3, 3), (2, 3)])])
    return gpd.GeoDataFrame(
        {
            "id": [1, 2, 3, 4, 5],
            "geometry": [
                Point(0, 0),
                Point(1, 1),
                poly,
                multipoly,
                LineString([(0, 0), (1, 1)]),
            ],
        },
        crs="EPSG:4326",
    )


def test_filter_points():
    gdf = _mixed_gdf()
    result = filter_by_geometry_type(gdf, geometry_types=["Point"])
    assert isinstance(result, gpd.GeoDataFrame)
    assert list(result["id"].values) == [1, 2]
    assert (result.geometry.geom_type == "Point").all()


def test_filter_polygons_and_multipolygons():
    gdf = _mixed_gdf()
    result = filter_by_geometry_type(gdf, geometry_types=["Polygon", "MultiPolygon"])
    assert list(result["id"].values) == [3, 4]
    assert set(result.geometry.geom_type) == {"Polygon", "MultiPolygon"}


def test_filter_no_match_returns_empty():
    gdf = _mixed_gdf()
    result = filter_by_geometry_type(gdf, geometry_types=["GeometryCollection"])
    assert isinstance(result, gpd.GeoDataFrame)
    assert len(result) == 0


def test_filter_empty_list_returns_empty():
    gdf = _mixed_gdf()
    result = filter_by_geometry_type(gdf, geometry_types=[])
    assert len(result) == 0


def test_filter_preserves_crs_and_columns():
    gdf = _mixed_gdf()
    result = filter_by_geometry_type(gdf, geometry_types=["Point"])
    assert result.crs == gdf.crs
    assert list(result.columns) == list(gdf.columns)


def test_filter_on_empty_gdf():
    gdf = gpd.GeoDataFrame({"id": [], "geometry": []}, crs="EPSG:4326")
    result = filter_by_geometry_type(gdf, geometry_types=["Point"])
    assert len(result) == 0
