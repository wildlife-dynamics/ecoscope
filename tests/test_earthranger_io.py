import datetime
import json
import uuid
from unittest.mock import MagicMock, patch

import geopandas as gpd
import pandas as pd
import pytest
import pytz
from erclient import ERClientException, ERClientNotFound
from shapely.geometry import Point

import ecoscope
from ecoscope.io.earthranger import EarthRangerIO
from ecoscope.io.utils import TIME_COLS

pytestmark = pytest.mark.io


EXPECTED_FEATURE_GROUP_COLUMNS = {
    "created_at",
    "updated_at",
    "feature_type",
    "name",
    "short_name",
    "external_id",
    "external_source",
    "description",
    "attributes",
    "provenance",
    "feature_geometry_webmercator",
    "spatialfile",
    "arcgis_item",
    "das_tenant",
    "pk",
}


def check_time_is_parsed(df):
    for col in TIME_COLS:
        if col in df.columns:
            assert pd.api.types.is_datetime64_ns_dtype(df[col]) or df[col].isna().all()


def test_get_subject_observations(er_io):
    relocations = er_io.get_subject_observations(
        subject_ids=er_io.SUBJECT_IDS,
        include_subject_details=True,
        include_source_details=True,
        include_subjectsource_details=True,
    )
    assert not relocations.gdf.empty
    assert isinstance(relocations, ecoscope.Relocations)
    assert "groupby_col" in relocations.gdf
    assert "fixtime" in relocations.gdf
    assert "extra__source" in relocations.gdf
    check_time_is_parsed(relocations.gdf)


def test_get_source_observations(er_io):
    relocations = er_io.get_source_observations(
        source_ids=er_io.SOURCE_IDS,
        include_source_details=True,
    )
    assert isinstance(relocations, ecoscope.Relocations)
    assert "fixtime" in relocations.gdf
    assert "groupby_col" in relocations.gdf
    check_time_is_parsed(relocations.gdf)


def test_get_source_no_observations(er_io):
    relocations = er_io.get_source_observations(
        source_ids=str(uuid.uuid4()),
        include_source_details=True,
    )
    assert relocations.gdf.empty


def test_get_subjectsource_observations(er_io):
    relocations = er_io.get_subjectsource_observations(
        subjectsource_ids=er_io.SUBJECTSOURCE_IDS,
        include_source_details=True,
    )
    assert isinstance(relocations, ecoscope.Relocations)
    assert "fixtime" in relocations.gdf
    assert "groupby_col" in relocations.gdf
    check_time_is_parsed(relocations.gdf)


def test_get_subjectsource_no_observations(er_io):
    relocations = er_io.get_subjectsource_observations(
        subjectsource_ids=str(uuid.uuid4()),
        include_source_details=True,
    )
    assert relocations.gdf.empty


def test_get_subjectgroup_observations(er_io):
    relocations = er_io.get_subjectgroup_observations(subject_group_name=er_io.GROUP_NAME)
    assert "groupby_col" in relocations.gdf
    assert len(relocations.gdf["extra__subject_id"].unique()) == 2


def test_get_events(er_events_io):
    events = er_events_io.get_events(event_type=["e00ce1f6-f9f1-48af-93c9-fb89ec493b8a"])
    assert not events.empty
    check_time_is_parsed(events)


def test_das_client_method(er_io):
    er_io.pulse()
    er_io.get_me()


def test_get_patrols_datestr(er_io):
    since_str = "2017-01-01"
    since_time = pd.to_datetime(since_str).replace(tzinfo=pytz.UTC)
    until_str = "2017-04-01"
    until_time = pd.to_datetime(until_str).replace(tzinfo=pytz.UTC)
    patrols = er_io.get_patrols(since=since_str, until=until_str)

    assert len(patrols) > 0
    check_time_is_parsed(patrols)

    time_ranges = [
        segment["time_range"]
        for segments in patrols["patrol_segments"]
        for segment in segments
        if "time_range" in segment
    ]

    for time_range in time_ranges:
        start = pd.to_datetime(time_range["start_time"])
        end = pd.to_datetime(time_range["end_time"])

        assert start <= until_time and end >= since_time


def test_get_patrols_datestr_invalid_format(er_io):
    with pytest.raises(ValueError):
        er_io.get_patrols(since="not a date")


def test_get_patrols_with_type_value(er_io):
    patrols = er_io.get_patrols(since="2017-01-01", until="2017-04-01", patrol_type_value="ecoscope_patrol")

    patrol_types = [
        segment["patrol_type"]
        for segments in patrols["patrol_segments"]
        for segment in segments
        if "patrol_type" in segment
    ]
    assert all(value == "ecoscope_patrol" for value in patrol_types)
    check_time_is_parsed(patrols)


def test_get_patrols_with_type_value_list(er_io):
    patrol_type_value_list = ["ecoscope_patrol", "MEP_Distance_Survey_Patrol"]
    patrols = er_io.get_patrols(since="2024-01-01", until="2024-04-01", patrol_type_value=patrol_type_value_list)

    patrol_types = [
        segment["patrol_type"]
        for segments in patrols["patrol_segments"]
        for segment in segments
        if "patrol_type" in segment
    ]
    assert all(value in patrol_type_value_list for value in patrol_types)
    check_time_is_parsed(patrols)


def test_get_patrols_with_invalid_type_value(er_io):
    with pytest.raises(ValueError):
        er_io.get_patrols(since="2017-01-01", until="2017-04-01", patrol_type_value="invalid")


def test_get_patrol_events(er_io):
    events = er_io.get_patrol_events(
        since=pd.Timestamp("2017-01-01").isoformat(),
        until=pd.Timestamp("2017-04-01").isoformat(),
    )
    assert "id" in events
    assert "event_type" in events
    assert "geometry" in events
    assert "patrol_id" in events
    assert "patrol_segment_id" in events
    assert "patrol_start_time" in events
    assert "patrol_subject" in events
    assert "patrol_type" in events
    assert "patrol_serial_number" in events
    assert "time" in events
    check_time_is_parsed(events)


def test_get_patrol_events_with_event_type_filter(er_io):
    event_type_filter = ["hwc_rep", "fire_rep"]
    events = er_io.get_patrol_events(
        since=pd.Timestamp("2017-01-01").isoformat(),
        until=pd.Timestamp("2017-04-01").isoformat(),
        event_type=event_type_filter,
    )
    assert not events.empty
    assert len(events["event_type"].unique().tolist()) == len(event_type_filter)


@patch("ecoscope.io.EarthRangerIO.get_patrols")
def test_get_patrol_events_empty(patrols_mock, er_io):
    patrols_mock.return_value = pd.DataFrame()

    events = er_io.get_patrol_events(
        since=pd.Timestamp("2017-01-01").isoformat(),
        until=pd.Timestamp("2017-04-01").isoformat(),
    )
    assert events.empty


def test_post_sourceproviders(er_io):
    response = er_io.post_sourceproviders(provider_key="test_provider_key", display_name="Test Provider Key")
    assert "id" in response
    assert "provider_key" in response
    assert "display_name" in response


def test_post_observations(er_io):
    observations = gpd.GeoDataFrame.from_dict(
        [
            {
                "recorded_at": pd.Timestamp.utcnow().isoformat(),
                "geometry": Point(0, 0),
                "source": er_io.SOURCE_IDS[0],
            },
            {
                "recorded_at": (pd.Timestamp.utcnow() + pd.Timedelta(seconds=1)).isoformat(),
                "geometry": Point(1, 1),
                "source": er_io.SOURCE_IDS[1],
            },
        ]
    )

    response = er_io.post_observations(observations)
    assert len(response) == 2
    assert "location" in response
    assert "recorded_at" in response


def test_post_events(er_io):
    events = [
        {
            "id": str(uuid.uuid4()),
            "title": "Accident",
            "event_type": "accident_rep",
            "time": pd.Timestamp.utcnow(),
            "location": {"latitude": -2.9553841592697982, "longitude": 38.033294677734375},
            "priority": 200,
            "state": "new",
            "event_details": {"type_accident": "head-on collision", "number_people_involved": 3, "animals_involved": 1},
            "is_collection": False,
            "icon_id": "accident_rep",
        },
        {
            "id": str(uuid.uuid4()),
            "title": "Accident",
            "event_type": "accident_rep",
            "time": pd.Timestamp.utcnow(),
            "location": {"latitude": -3.0321834919139206, "longitude": 38.4906005859375},
            "priority": 300,
            "state": "active",
            "event_details": {
                "type_accident": "side-impact collision",
                "number_people_involved": 2,
                "animals_involved": 1,
            },
            "is_collection": False,
            "icon_id": "accident_rep",
        },
    ]
    results = er_io.post_event(events)
    results["time"] = pd.to_datetime(results["time"], utc=True)

    expected = pd.DataFrame(events)
    results = results[expected.columns]
    pd.testing.assert_frame_equal(results, expected)


def test_patch_event(er_io):
    event = [
        {
            "id": str(uuid.uuid4()),
            "title": "Arrest",
            "event_type": "arrest_rep",
            "time": pd.Timestamp.utcnow(),
            "location": {"latitude": -3.4017015747197306, "longitude": 38.11809539794921},
            "priority": 200,
            "state": "new",
            "event_details": {
                "arrestrep_dateofbirth": "1985-01-1T13:00:00.000Z",
                "arrestrep_nationality": "other",
                "arrestrep_timeofarrest": datetime.datetime.utcnow().isoformat(),
                "arrestrep_reaonforarrest": "firearm",
                "arrestrep_arrestingranger": "catherine's cellphone",
            },
            "is_collection": False,
            "icon_id": "arrest_rep",
        }
    ]
    er_io.post_event(event)
    event_id = event[0]["id"]

    updated_event = pd.DataFrame(
        [
            {
                "priority": 300,
                "state": "active",
                "location": {"latitude": -4.135503657998179, "longitude": 38.4576416015625},
            }
        ]
    )

    result = er_io.patch_event(event_id=event_id, events=updated_event)
    result = result[["priority", "state", "location"]]
    pd.testing.assert_frame_equal(result, updated_event)


def test_get_patrol_observations(er_io):
    patrols = er_io.get_patrols(
        since=pd.Timestamp("2024-01-01").isoformat(),
        until=pd.Timestamp("2024-12-01").isoformat(),
    )

    observations = er_io.get_patrol_observations(
        patrols,
        include_source_details=False,
        include_subject_details=False,
        include_subjectsource_details=False,
    )
    assert not observations.gdf.empty
    check_time_is_parsed(observations.gdf)


def test_get_patrol_observations_with_patrol_details(er_io):
    patrols = er_io.get_patrols(
        since=pd.Timestamp("2017-01-01").isoformat(),
        until=pd.Timestamp("2017-04-01").isoformat(),
    )

    observations = er_io.get_patrol_observations(
        patrols,
        include_source_details=False,
        include_subject_details=False,
        include_subjectsource_details=False,
        include_patrol_details=True,
    )
    assert not observations.gdf.empty
    assert "patrol_id" in observations.gdf.columns
    assert "patrol_title" in observations.gdf.columns
    assert "patrol_type" in observations.gdf.columns
    assert "patrol_status" in observations.gdf.columns
    assert "patrol_subject" in observations.gdf.columns
    pd.testing.assert_series_equal(observations.gdf["patrol_id"], observations.gdf["groupby_col"], check_names=False)
    check_time_is_parsed(observations.gdf)


def test_users(er_io):
    users = pd.DataFrame(er_io.get_users())
    assert not users.empty


def test_get_spatial_feature(er_io):
    spatial_feature = er_io.get_spatial_feature(spatial_feature_id="8868718f-0154-45bf-a74d-a66706ef958f")
    assert not spatial_feature.empty


def test_get_spatial_features_group(er_io):
    sfg = er_io.get_spatial_features_group(spatial_features_group_id="15698426-7e0f-41df-9bc3-495d87e2e097")
    assert isinstance(sfg, gpd.GeoDataFrame)
    assert sfg.crs == "EPSG:4326"
    assert not sfg.empty
    assert EXPECTED_FEATURE_GROUP_COLUMNS.issubset(sfg.columns)


def test_get_spatial_features_group_with_group_data(er_io):
    group_name = "mep"
    group_id = "15698426-7e0f-41df-9bc3-495d87e2e097"

    sfg = er_io.get_spatial_features_group(spatial_features_group_id=group_id, with_group_data=True)
    assert isinstance(sfg, dict)
    assert {"id", "name", "description", "url", "features"}.issubset(sfg.keys())
    assert sfg["name"] == group_name
    assert isinstance(sfg["features"], gpd.GeoDataFrame)
    assert not sfg["features"].empty
    assert EXPECTED_FEATURE_GROUP_COLUMNS.issubset(sfg["features"].columns)


@pytest.fixture
def mock_spatial_feature_group():
    return {
        "id": "1234",
        "name": "Test",
        "description": "",
        "url": "https://mep-dev.pamdas.org/api/v1.0/spatialfeaturegroup/1234",
        "features": [
            {
                "type": "FeatureCollection",
                "crs": {"type": "name", "properties": {}},
                "features": [
                    {
                        "type": "Feature",
                        "properties": {
                            "created_at": "2024-08-14T13:07:02.787Z",
                            "updated_at": "2024-08-15T09:39:48.797Z",
                            "feature_type": "45678",
                            "name": "Test Feature",
                            "short_name": "TF",
                            "pk": "7777",
                        },
                        "geometry": {
                            "type": "MultiLineString",
                            "coordinates": [
                                [28.350274200053384, -15.43285045203909],
                                [28.350122373365377, -15.432920120084994],
                                [28.350037551097166, -15.433094419501986],
                                [28.3500412620714, -15.433315228934031],
                                [28.350093500175603, -15.43354090013023],
                            ],
                        },
                    }
                ],
            }
        ],
    }


@patch("erclient.client.ERClient._get")
def test_get_spatial_features_group_with_no_crs_raises(er_mock, er_io, mock_spatial_feature_group):
    er_mock.return_value = mock_spatial_feature_group
    with pytest.raises(ValueError, match="CRS information missing for spatial feature group 1234"):
        er_io.get_spatial_features_group(spatial_features_group_id=1234)


def test_get_subjects_chunking(er_io):
    subject_ids = ",".join(er_io.SUBJECT_IDS)
    single_request_result = er_io.get_subjects(id=subject_ids)
    chunked_request_result = er_io.get_subjects(id=subject_ids, max_ids_per_request=1)

    pd.testing.assert_frame_equal(single_request_result, chunked_request_result)


def test_existing_token(er_io):
    new_client = ecoscope.io.EarthRangerIO(
        service_root=er_io.service_root, token_url=er_io.token_url, token=er_io.auth.get("access_token")
    )

    events = new_client.get_patrol_events(
        since=pd.Timestamp("2017-01-01").isoformat(),
        until=pd.Timestamp("2017-04-01").isoformat(),
    )
    assert not events.empty


def test_existing_token_expired(er_io):
    token = er_io.auth.get("access_token")
    er_io.refresh_token()

    with pytest.raises(ERClientException, match="Authorization token is invalid or expired."):
        ecoscope.io.EarthRangerIO(service_root=er_io.service_root, token_url=er_io.token_url, token=token)


def test_get_patrol_observations_with_patrol_filter(er_io):
    observations = er_io.get_patrol_observations_with_patrol_filter(
        since=pd.Timestamp("2017-01-01").isoformat(),
        until=pd.Timestamp("2017-04-01").isoformat(),
        patrol_type_value="ecoscope_patrol",
        status=["done"],
        include_patrol_details=True,
    )

    assert not observations.gdf.empty
    assert "patrol_id" in observations.gdf.columns
    assert "patrol_title" in observations.gdf.columns
    assert "patrol_start_time" in observations.gdf.columns
    assert "patrol_type" in observations.gdf.columns
    pd.testing.assert_series_equal(observations.gdf["patrol_id"], observations.gdf["groupby_col"], check_names=False)
    check_time_is_parsed(observations.gdf)


@patch("erclient.client.ERClient.get_objects_multithreaded")
def test_get_events_bad_geojson(get_objects_mock, sample_events_df_with_bad_geojson, er_io):
    get_objects_mock.return_value = sample_events_df_with_bad_geojson

    events = er_io.get_events(event_type=["e00ce1f6-f9f1-48af-93c9-fb89ec493b8a"])
    assert not events.empty
    # of the 6 id's in the mock we expect these 4 to be returned
    assert events.index.to_list() == [
        "bcda9c6a-628c-4825-947d-72f66115fc09",
        "d464672a-3cc2-4d9a-bb3f-a69c34efb09c",
        "4a599a57-7a89-4eb3-bb11-d2a36d1627e2",
        "bcb01505-c635-48eb-b176-2b1390a0a5bf",
    ]

    events_with_null_geoms = er_io.get_events(
        event_type=["e00ce1f6-f9f1-48af-93c9-fb89ec493b8a"], drop_null_geometry=False
    )
    assert len(events_with_null_geoms) == 6


@patch("erclient.client.ERClient.get_objects_multithreaded")
def test_get_patrol_events_bad_geojson(get_objects_mock, sample_patrol_events_with_bad_geojson, er_io):
    get_objects_mock.return_value = sample_patrol_events_with_bad_geojson

    patrol_events = er_io.get_patrol_events(
        since=pd.Timestamp("2017-01-01").isoformat(),
        until=pd.Timestamp("2017-04-01").isoformat(),
    )
    assert not patrol_events.empty
    # By default, we're rejecting any geojson that's missing geometry or a timestamp
    assert patrol_events.id.to_list() == ["ebf812f5-e616-40e4-8fcf-ebb3ef6a6364"]

    patrol_events_with_null_geoms = er_io.get_patrol_events(
        since=pd.Timestamp("2017-01-01").isoformat(),
        until=pd.Timestamp("2017-04-01").isoformat(),
        drop_null_geometry=False,
    )
    assert len(patrol_events_with_null_geoms) == 2


@patch("erclient.client.ERClient.get_objects_multithreaded")
def test_get_patrol_events_mixed_geom(get_objects_mock, sample_patrol_events_with_poly, er_io):
    get_objects_mock.return_value = sample_patrol_events_with_poly

    patrol_events_mixed = er_io.get_patrol_events(
        since=pd.Timestamp("2017-01-01").isoformat(),
        until=pd.Timestamp("2017-04-01").isoformat(),
        force_point_geometry=False,
    )
    assert not patrol_events_mixed.empty
    assert len(patrol_events_mixed.geom_type.unique()) == 3

    patrol_events_points = er_io.get_patrol_events(
        since=pd.Timestamp("2017-01-01").isoformat(),
        until=pd.Timestamp("2017-04-01").isoformat(),
        force_point_geometry=True,
    )
    assert not patrol_events_points.empty
    assert len(patrol_events_points.geom_type.unique()) == 1
    assert patrol_events_points.geom_type[0] == "Point"


@pytest.mark.parametrize(
    "er_callable, er_kwargs",
    [
        (EarthRangerIO.get_patrols, {}),
        (EarthRangerIO.get_subjectgroup_observations, {"subject_group_id": "12345"}),
        (EarthRangerIO.get_patrol_observations_with_patrol_filter, {}),
        (EarthRangerIO.get_patrol_events, {}),
        (EarthRangerIO.get_events, {}),
    ],
)
@patch("erclient.client.ERClient._get")
def test_empty_responses(_get_mock, er_io, er_callable, er_kwargs):
    _get_mock.return_value = {}
    df = er_callable(er_io, **er_kwargs)
    assert df.empty


@pytest.mark.parametrize(
    "er_callable, er_kwargs",
    [
        (EarthRangerIO.get_events, {}),
        (EarthRangerIO.get_source_observations, {"source_ids": ["12345"]}),
    ],
)
@pytest.mark.parametrize("set_sub_page_size", [True, False])
@patch("erclient.client.ERClient.get_objects_multithreaded")
def test_page_size_override(get_objects_mock, set_sub_page_size, er_callable, er_kwargs, er_io):
    get_objects_mock.return_value = {}
    if set_sub_page_size:
        er_kwargs["sub_page_size"] = 100

    er_callable(er_io, **er_kwargs)

    for call in get_objects_mock.mock_calls:
        assert call.kwargs["page_size"] == 100 if set_sub_page_size else 4000


@pytest.mark.parametrize(
    "er_callable, er_kwargs",
    [
        (EarthRangerIO.get_patrols, {}),
        (EarthRangerIO.get_patrol_observations_with_patrol_filter, {}),
        (EarthRangerIO.get_patrol_events, {}),
    ],
)
@pytest.mark.parametrize("set_sub_page_size", [True, False])
@patch("ecoscope.io.earthranger.EarthRangerIO.get_objects_multithreaded")
def test_page_size_override_with_patrols(get_objects_mock, set_sub_page_size, er_callable, er_kwargs, er_io):
    """
    The intent of this test is to make sure that if page size is set on the er_callable,
    _all_ downstream calls to get_objects_multithreaded use the overridden page_size
    """
    patrols_json = json.loads(open("tests/sample_data/io/sample_patrol.json", "r").read())

    def patrols_side_effect(*args, **kwargs):
        yield patrols_json if get_objects_mock.call_count == 1 else MagicMock()

    get_objects_mock.side_effect = patrols_side_effect

    if set_sub_page_size:
        er_kwargs["sub_page_size"] = 100

    er_callable(er_io, **er_kwargs)

    for call in get_objects_mock.mock_calls:
        assert call.kwargs["page_size"] == 100 if set_sub_page_size else 4000


@pytest.mark.parametrize(
    "er_callable, er_kwargs",
    [
        (EarthRangerIO.get_subjectgroup_observations, {"subject_group_id": "12345"}),
    ],
)
@pytest.mark.parametrize("set_sub_page_size", [True, False])
@patch("ecoscope.io.earthranger.EarthRangerIO.get_objects_multithreaded")
def test_page_size_override_with_subjects(get_objects_mock, set_sub_page_size, er_callable, er_kwargs, er_io):
    """
    The intent of this test is to make sure that if page size is set on the er_callable,
    _all_ downstream calls to get_objects_multithreaded use the overridden page_size
    """
    subjects_json = json.loads(open("tests/sample_data/io/sample_subject.json", "r").read())
    observation_json = json.loads(open("tests/sample_data/io/sample_observation.json", "r").read())

    def patrols_side_effect(*args, **kwargs):
        yield subjects_json
        yield observation_json

    get_objects_mock.side_effect = patrols_side_effect

    if set_sub_page_size:
        er_kwargs["sub_page_size"] = 100

    er_callable(er_io, **er_kwargs)

    for call in get_objects_mock.mock_calls:
        assert call.kwargs["page_size"] == 100 if set_sub_page_size else 4000


def test_get_patrols_page_size_parity(er_io):
    kwargs = {"since": "2017-01-01", "until": "2017-04-01", "patrol_type_value": "ecoscope_patrol"}
    patrols_default = er_io.get_patrols(**kwargs)
    patrols_page_size = er_io.get_patrols(**(kwargs | {"sub_page_size": 100}))
    pd.testing.assert_frame_equal(patrols_default, patrols_page_size)


def test_get_subjectgroup_observations_page_size_parity(er_io):
    kwargs = {
        "subject_group_name": er_io.GROUP_NAME,
        "since": "2014-01-01",
        "until": "2014-06-01",
    }
    relocations_default = er_io.get_subjectgroup_observations(**kwargs)
    relocations_page_size = er_io.get_subjectgroup_observations(**(kwargs | {"sub_page_size": 100}))
    pd.testing.assert_frame_equal(relocations_default.gdf, relocations_page_size.gdf)


def test_get_patrol_events_page_size_parity(er_io):
    kwargs = {
        "since": pd.Timestamp("2017-01-01").isoformat(),
        "until": pd.Timestamp("2017-04-01").isoformat(),
    }
    events_default = er_io.get_patrol_events(**kwargs)
    events_page_size = er_io.get_patrol_events(**(kwargs | {"sub_page_size": 100}))
    pd.testing.assert_frame_equal(events_default, events_page_size)


def test_get_patrol_observations_with_patrol_filter_page_size_parity(er_io):
    kwargs = {
        "since": pd.Timestamp("2017-01-01").isoformat(),
        "until": pd.Timestamp("2017-04-01").isoformat(),
        "patrol_type_value": "ecoscope_patrol",
        "status": ["done"],
        "include_patrol_details": True,
    }

    relocations_default = er_io.get_patrol_observations_with_patrol_filter(**kwargs)
    relocations_page_size = er_io.get_patrol_observations_with_patrol_filter(**(kwargs | {"sub_page_size": 100}))
    pd.testing.assert_frame_equal(relocations_default.gdf, relocations_page_size.gdf)


def test_get_events_page_size_parity(er_io):
    kwargs = {
        "since": pd.Timestamp("2017-01-01").isoformat(),
        "until": pd.Timestamp("2017-04-01").isoformat(),
    }
    events_default = er_io.get_events(**kwargs)
    events_page_size = er_io.get_events(**(kwargs | {"sub_page_size": 100}))
    # get_events explicitly sets the index as the event id
    # but keeps the original range index, so drop that here
    # check_like = True since the sort order of the events can differ
    pd.testing.assert_frame_equal(
        events_default.drop(columns=["index"]), events_page_size.drop(columns=["index"]), check_like=True
    )


@pytest.mark.parametrize(
    "api_version",
    ["v1", "v2", "both"],
)
def test_get_event_types_api_version(er_events_io, api_version):
    known_v1_event_type = "fe77a01d-a11f-4608-b6c6-2caf6f0b838d"
    known_v2_event_type = "8081cb2c-e145-41ad-9726-b8f9d1ce3907"
    event_types = er_events_io.get_event_types(api_version=api_version)

    assert not event_types.empty
    if api_version == "v1" or api_version == "both":
        assert known_v1_event_type in event_types.id.values
    if api_version == "v2" or api_version == "both":
        assert known_v2_event_type in event_types.id.values


def test_get_event_types_response_shape(er_events_io):
    v1_event_types = er_events_io.get_event_types(api_version="v1")
    v2_event_types = er_events_io.get_event_types(api_version="v2")

    assert not v1_event_types.empty
    assert not v2_event_types.empty
    assert set(v1_event_types.columns) == set(v2_event_types.columns)


def test_get_events_in_chunks(er_events_io):
    all_event_types = er_events_io.get_event_types()
    assert len(all_event_types) > ecoscope.io.earthranger.SAFE_QUERY_PARAM_LIST_SIZE

    events = er_events_io.get_events(
        since=pd.Timestamp("2017-01-01").isoformat(),
        until=pd.Timestamp("2017-04-01").isoformat(),
        event_type=all_event_types["id"].values,
    )

    assert len(events) > 0
    assert "id" == events.index.name
    assert "time" in events
    assert "event_type" in events
    assert "geometry" in events
    assert len(events["event_type"].unique()) > 1


def test_get_events_empty_event_types(er_events_io):
    events = er_events_io.get_events(
        since=pd.Timestamp("2017-01-01").isoformat(),
        until=pd.Timestamp("2017-04-01").isoformat(),
    )
    assert not events.empty
    check_time_is_parsed(events)


@pytest.fixture(scope="session")
def sample_events(er_events_io) -> gpd.GeoDataFrame:
    return er_events_io.get_events(
        since=pd.Timestamp("2025-10-01").isoformat(),
        until=pd.Timestamp("2025-10-31").isoformat(),
    ).copy()


def test_get_event_type_display_names_from_events_no_categories(er_events_io, sample_events):
    events = er_events_io.get_event_type_display_names_from_events(
        events_gdf=sample_events,
        append_category_names="never",
    )

    assert not events.empty
    assert "event_type", "event_type_display" in events
    assert not events["event_type_display"].isna().any()
    assert "Accident" in events["event_type_display"].unique()
    assert "Arrest" in events["event_type_display"].unique()


def test_get_event_type_display_names_from_events_all_categories(er_events_io, sample_events):
    events = er_events_io.get_event_type_display_names_from_events(
        events_gdf=sample_events,
        append_category_names="always",
    )

    assert not events.empty
    assert "event_type", "event_type_display" in events
    assert not events["event_type_display"].isna().any()
    assert "Accident (Security)" in events["event_type_display"].unique()
    assert "Arrest (Security)" in events["event_type_display"].unique()


def test_get_event_type_display_names_from_events_categories_duplicates_only(er_events_io, sample_events):
    events = er_events_io.get_event_type_display_names_from_events(
        events_gdf=sample_events,
        append_category_names="duplicates",
    )

    assert not events.empty
    assert "event_type", "event_type_display" in events
    assert not events["event_type_display"].isna().any()
    assert "Accident" in events["event_type_display"].unique()
    assert "Arrest" in events["event_type_display"].unique()
    assert "Test Event (Logistics)" in events["event_type_display"].unique()
    assert "Test Event (Monitoring)" in events["event_type_display"].unique()
    assert "Test Event (Human Wildlife Conflict)" in events["event_type_display"].unique()
    assert "Inactive Event" in events["event_type_display"].unique()


def test_get_event_type_display_names_from_patrol_events(er_events_io):
    patrol_events = er_events_io.get_patrol_events(
        since=pd.Timestamp("2017-01-01").isoformat(),
        until=pd.Timestamp("2017-04-01").isoformat(),
    )

    events = er_events_io.get_event_type_display_names_from_events(
        events_gdf=patrol_events,
        append_category_names="always",
    )

    assert not events.empty
    assert "event_type", "event_type_display" in events
    assert not events["event_type_display"].isna().any()
    assert "Poachers Camp (Security)" in events["event_type_display"].unique()
    assert "Fire (Monitoring)" in events["event_type_display"].unique()


def test_get_choices_from_v2_event_type(er_io):
    choices = er_io.get_choices_from_v2_event_type(event_type="elephant_sigthing_test", choice_field="herd_type")
    assert isinstance(choices, dict)
    assert {"bull_only": "Bull Only", "female_only": "Female Only", "mix": "Mix"} == choices


def test_get_choices_from_v2_event_type_non_existent_choice_field(er_io):
    choices = er_io.get_choices_from_v2_event_type(event_type="elephant_sigthing_test", choice_field=" ")
    assert choices == {}


def test_get_choices_from_v2_event_type_non_existent_event_type(er_io):
    with pytest.raises(ERClientNotFound):
        er_io.get_choices_from_v2_event_type(event_type=" ", choice_field="herd_type")


def test_get_fields_from_v2_event_type(er_io):
    fields = er_io.get_fields_from_event_type_schema(event_type="elephant_sigthing_test")
    assert isinstance(fields, dict)
    assert {
        "name_of_collared_elephant": "Name of collared elephant",
        "herd_type": "Herd Type",
        "how_many_identifiable_individuals": "How many identifiable individuals?",
        "how_many_elephants_are_in_the_group": "How many elephants are in the group?",
    } == fields


def test_get_fields_from_v1_event_type(er_io):
    fields = er_io.get_fields_from_event_type_schema(event_type="accident_rep")
    assert isinstance(fields, dict)
    assert {
        "type_accident": "Type of accident",
        "number_people_involved": "Number of people involved",
        "animals_involved": "Animals involved",
    } == fields


def test_get_fields_from_non_existent_event_type(er_io):
    with pytest.raises(ERClientNotFound):
        er_io.get_fields_from_event_type_schema(event_type=" ")
