import inspect
import pickle
import threading
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock, patch

import ecoscope.io.earthranger as er_module
from ecoscope.io.earthranger import EarthRangerIO

GEOM = {"type": "Point", "coordinates": [1, 2]}
V1 = "https://x.pamdas.org/api/v1.0"
V2 = "https://x.pamdas.org/api/v2.0"


def _io():
    er = EarthRangerIO.__new__(EarthRangerIO)
    er.server = "https://x.pamdas.org"
    er.service_root = V1
    return er


def _recording_io(urls):
    """A client whose requests, through erclient's real code, go to a fake session that records each URL."""
    er = _io()
    er.user_agent = "test"
    er.auth_headers = lambda: {}

    def send(url, **kwargs):
        urls.append(url)
        threading.Event().wait(0.001)  # let other threads run mid-request
        return MagicMock(ok=True, text="{}", **{"json.return_value": {}})

    er._http_session = MagicMock(get=send, post=send, patch=send, delete=send)
    return er


def test_v2_scope_switches_urls_and_restores_v1():
    er = _io()
    assert er._er_url("feature/a/") == f"{V1}/feature/a/"
    with er._use_v2_api():
        assert er._er_url("feature/a/") == f"{V2}/feature/a/"
        assert er.service_root == V1  # shared state is never touched
    assert er._er_url("feature/a/") == f"{V1}/feature/a/"


def test_v2_scope_restores_v1_after_an_error():
    er = _io()
    try:
        with er._use_v2_api():
            raise RuntimeError
    except RuntimeError:
        pass
    assert er._er_url("x/") == f"{V1}/x/"


def test_nested_v2_scopes_stay_on_v2_until_the_outer_one_exits():
    er = _io()
    with er._use_v2_api():
        with er._use_v2_api():
            pass
        assert er._er_url("x/") == f"{V2}/x/"
    assert er._er_url("x/") == f"{V1}/x/"


def test_v2_scope_is_per_client():
    er, other = _io(), _io()
    with er._use_v2_api():
        assert other._er_url("x/") == f"{V1}/x/"


def test_v2_scope_is_per_thread():
    er = _io()
    inside, done = threading.Event(), threading.Event()

    def hold_v2_scope():
        with er._use_v2_api():
            inside.set()
            done.wait(5)

    worker = threading.Thread(target=hold_v2_scope)
    worker.start()
    inside.wait(5)
    try:
        assert er._er_url("x/") == f"{V1}/x/"  # this thread is unaffected
    finally:
        done.set()
        worker.join()


def test_parallel_v2_writes_never_reach_v1():
    """The sync patches features from a thread pool on one shared client; every call must hit v2."""
    urls: list[str] = []
    er = _recording_io(urls)
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda i: er.update_spatial_feature(str(i), name="n"), range(200)))
    assert len(urls) == 200
    assert all(u.startswith(f"{V2}/feature/") for u in urls)


# Every HTTP method, half on v2 and half on v1, keyed by the path each call requests.
_MIXED_CALLS = {
    "features/": lambda er: er.create_spatial_feature("n", "t", GEOM, {}),  # POST v2
    "feature/f/": lambda er: er.update_spatial_feature("f", name="n"),  # PATCH v2
    "feature/d/": lambda er: er.delete_spatial_feature("d"),  # DELETE v2
    "activity/eventtypes": lambda er: er.get_event_types(api_version="v2"),  # GET v2
    "featuretypes/": lambda er: er.create_feature_type("t"),  # POST v1
    "featuregroup/g/": lambda er: er.set_feature_group_members("g", []),  # PATCH v1
    "featuregroups": lambda er: er.get_feature_groups(),  # GET v1
    "activity/events/eventtypes": lambda er: er.get_event_types(api_version="v1"),  # GET v1
}
_V2_PATHS = {"features/", "feature/f/", "feature/d/", "activity/eventtypes"}


def test_every_method_keeps_its_version_under_heavy_concurrency():
    urls: list[str] = []
    er = _recording_io(urls)
    calls = list(_MIXED_CALLS.values()) * 250
    with ThreadPoolExecutor(max_workers=16) as pool:
        list(pool.map(lambda call: call(er), calls))

    assert len(urls) == len(calls)
    expected = {f"{V2 if p in _V2_PATHS else V1}/{p}" for p in _MIXED_CALLS}
    assert set(urls) == expected  # nothing landed on the wrong version


def test_event_types_from_both_versions():
    urls: list[str] = []
    _recording_io(urls).get_event_types(api_version="both")
    assert urls == [f"{V1}/activity/events/eventtypes", f"{V2}/activity/eventtypes"]


def test_client_still_pickles_after_using_v2():
    er = _io()
    with er._use_v2_api():
        copy = pickle.loads(pickle.dumps(er))
    assert copy._er_url("x/") == f"{V1}/x/"  # a copy is a new client, outside the scope
    with copy._use_v2_api():
        assert copy._er_url("x/") == f"{V2}/x/"


def _called_in_v2(er):
    """A side effect recording whether the mocked request was made inside the v2 scope."""
    seen = []

    def record(*args, **kwargs):
        seen.append(er._er_url("").startswith(V2))
        return {"id": "n", "features": [{"type": "Feature", "geometry": GEOM, "properties": {"a": 1}}]}

    return seen, record


def test_create_spatial_feature_posts_v2_body():
    er = _io()
    seen, record = _called_in_v2(er)
    with patch.object(EarthRangerIO, "_post", side_effect=record) as post:
        out = er.create_spatial_feature("AV61", "t1", GEOM, {"a": 1})
    assert out["id"] == "n" and seen == [True]
    post.assert_called_once_with(
        "features/",
        {"name": "AV61", "feature_type": "t1", "feature_geometry": GEOM, "description": "", "attributes": {"a": 1}},
    )


def test_get_spatial_feature_uses_v2_feature_path():
    er = _io()
    seen, record = _called_in_v2(er)
    with patch.object(EarthRangerIO, "_get", side_effect=record) as get:
        out = er.get_spatial_feature("x")
    assert get.call_args.args[0] == "feature/x/" and seen == [True]
    assert len(out) == 1


def test_update_and_delete_use_v2_feature_path():
    er = _io()
    seen, record = _called_in_v2(er)
    with patch.object(EarthRangerIO, "_patch", side_effect=record) as patch_:
        er.update_spatial_feature("f1", name="N")
    patch_.assert_called_once_with("feature/f1/", {"name": "N"})
    with patch.object(EarthRangerIO, "_delete", side_effect=record) as delete:
        assert er.delete_spatial_feature("f1") is None
    delete.assert_called_once_with("feature/f1/")
    assert seen == [True, True]


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
