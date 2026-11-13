from unittest.mock import MagicMock, patch

import geopandas as gpd
import pytest
from shapely.geometry import Point

from ecoscope.io.earthranger import EarthRangerIO


def _io():
    return EarthRangerIO.__new__(EarthRangerIO)


def _features(n):
    return gpd.GeoDataFrame(geometry=[Point(i, i) for i in range(n)], crs="EPSG:4326")


def test_get_spatial_features_groups_fetches_data_for_each_group():
    er = _io()
    summaries = [
        {"id": "a", "name": "Blocks", "feature_count": 2},
        {"id": "b", "name": "Sectors", "feature_count": 1},
    ]
    with (
        patch.object(EarthRangerIO, "_get", return_value=summaries) as get,
        patch.object(
            EarthRangerIO,
            "get_spatial_features_group",
            side_effect=lambda **kw: {
                "id": kw["spatial_features_group_id"],
                "name": {"a": "Blocks", "b": "Sectors"}[kw["spatial_features_group_id"]],
                "features": _features(2),
            },
        ) as single,
    ):
        groups = er.get_spatial_features_groups()

    get.assert_called_once_with("spatialfeaturegroup")
    assert [g["id"] for g in groups] == ["a", "b"]
    assert all(isinstance(g["features"], gpd.GeoDataFrame) for g in groups)
    assert groups[0]["name"] == "Blocks"
    single.assert_any_call(spatial_features_group_id="a", with_group_data=True)


def test_get_spatial_features_groups_skips_empty_groups_and_handles_no_groups():
    er = _io()
    with patch.object(EarthRangerIO, "_get", return_value=[]):
        assert er.get_spatial_features_groups() == []

    summaries = [{"id": "a", "name": "Empty", "feature_count": 0}]
    single = MagicMock()
    with (
        patch.object(EarthRangerIO, "_get", return_value=summaries),
        patch.object(EarthRangerIO, "get_spatial_features_group", single),
    ):
        assert er.get_spatial_features_groups() == []
    single.assert_not_called()


def _collection(crs=True):
    fc = {
        "type": "FeatureCollection",
        "features": [{"type": "Feature", "properties": {}, "geometry": {"type": "Point", "coordinates": [1, 1]}}],
    }
    if crs:
        fc["crs"] = {"type": "name", "properties": {"name": "EPSG:4326"}}
    return fc


def test_get_spatial_features_groups_skips_a_group_it_cannot_read():
    er = _io()
    responses = {
        "spatialfeaturegroup": [
            {"id": "good", "name": "Blocks", "feature_count": 1},
            {"id": "hollow", "name": "Hollow", "feature_count": 3},
            {"id": "no-crs", "name": "No CRS"},
        ],
        "spatialfeaturegroup/good/": {"id": "good", "name": "Blocks", "features": [_collection()]},
        "spatialfeaturegroup/hollow/": {"id": "hollow", "name": "Hollow", "features": []},
        "spatialfeaturegroup/no-crs/": {"id": "no-crs", "name": "No CRS", "features": [_collection(crs=False)]},
    }
    with patch.object(EarthRangerIO, "_get", side_effect=lambda path, **kw: responses[path]):
        with pytest.warns(UserWarning) as caught:
            groups = er.get_spatial_features_groups()

    assert [g["id"] for g in groups] == ["good"]
    assert len(groups[0]["features"]) == 1
    messages = [str(w.message) for w in caught]
    assert any("Hollow" in m for m in messages)
    assert any("No CRS" in m for m in messages)
