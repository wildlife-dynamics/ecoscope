import inspect
from unittest.mock import patch

import pytest
import requests

import ecoscope.io.earthranger as er_module
from ecoscope.io.earthranger import EarthRangerIO

GEOM = {"type": "Point", "coordinates": [1, 2]}


def _io():
    return EarthRangerIO.__new__(EarthRangerIO)


@pytest.mark.parametrize(
    "method, path, expected",
    [
        ("POST", "../v2.0/features/", "https://x.pamdas.org/api/v2.0/features/"),
        ("PATCH", "../v2.0/feature/abc/", "https://x.pamdas.org/api/v2.0/feature/abc/"),
        ("DELETE", "../v2.0/feature/abc/", "https://x.pamdas.org/api/v2.0/feature/abc/"),
    ],
)
def test_v2_paths_resolve_to_api_v2(method, path, expected):
    er = _io()
    er.service_root = "https://x.pamdas.org/api/v1.0"
    assert requests.Request(method, er._er_url(path)).prepare().url == expected


def test_create_spatial_feature_posts_v2_body():
    er = _io()
    with patch.object(EarthRangerIO, "_post", return_value={"id": "n"}) as post:
        out = er.create_spatial_feature("AV61", "t1", GEOM, {"a": 1})
    assert out == {"id": "n"}
    post.assert_called_once_with(
        "../v2.0/features/",
        {"name": "AV61", "feature_type": "t1", "feature_geometry": GEOM, "description": "", "attributes": {"a": 1}},
    )


def test_update_and_delete_use_v2_feature_path():
    er = _io()
    with patch.object(EarthRangerIO, "_patch", return_value={}) as patch_:
        er.update_spatial_feature("f1", name="N")
    patch_.assert_called_once_with("../v2.0/feature/f1/", {"name": "N"})
    with patch.object(EarthRangerIO, "_delete") as delete:
        assert er.delete_spatial_feature("f1") is None
    delete.assert_called_once_with("../v2.0/feature/f1/")


def test_group_and_type_paths():
    er = _io()
    with patch.object(EarthRangerIO, "_get", return_value=[]) as get:
        er.get_feature_groups()
        get.assert_called_with("featuregroups")
        er.get_feature_types()
        get.assert_called_with("featuretypes")
        er.get_feature_group("g1")
        get.assert_called_with("featuregroup/g1/")
    with patch.object(EarthRangerIO, "_post", return_value={}) as post:
        er.create_feature_group("G")
        post.assert_called_with("featuregroups/", {"name": "G", "description": ""})
        er.create_feature_type("T")
        post.assert_called_with("featuretypes/", {"name": "T"})
    with patch.object(EarthRangerIO, "_patch", return_value={}) as patch_:
        er.set_feature_group_members("g1", ["a", "b"])
    patch_.assert_called_once_with("featuregroup/g1/", {"features": ["a", "b"]})


def test_member_ids_flatten_feature_collections():
    er = _io()
    group = {"features": [{"type": "FeatureCollection", "features": [{"id": "a"}, {"id": "b"}]}]}
    with patch.object(EarthRangerIO, "get_feature_group", return_value=group):
        assert er.get_feature_group_member_ids("g") == ["a", "b"]
    with patch.object(EarthRangerIO, "get_feature_group", return_value={"features": []}):
        assert er.get_feature_group_member_ids("g") == []


def test_no_deprecated_paths_left():
    source = inspect.getsource(er_module)
    assert "spatialfeature" not in source
    assert "featureclass" not in source
