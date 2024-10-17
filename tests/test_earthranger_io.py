import datetime
import uuid

import geopandas as gpd
import pandas as pd
import pytest
import pytz
from shapely.geometry import Point

import ecoscope
from erclient import ERClientException

if not pytest.earthranger:
    pytest.skip(
        "Skipping tests because connection to EarthRanger is not available.",
        allow_module_level=True,
    )


def test_get_subject_observations(er_io):
    relocations = er_io.get_subject_observations(
        subject_ids=er_io.SUBJECT_IDS,
        include_subject_details=True,
        include_source_details=True,
        include_subjectsource_details=True,
    )
    assert not relocations.empty
    assert isinstance(relocations, ecoscope.base.Relocations)
    assert "groupby_col" in relocations
    assert "fixtime" in relocations
    assert "extra__source" in relocations


def test_get_source_observations(er_io):
    relocations = er_io.get_source_observations(
        source_ids=er_io.SOURCE_IDS,
        include_source_details=True,
    )
    assert isinstance(relocations, ecoscope.base.Relocations)
    assert "fixtime" in relocations
    assert "groupby_col" in relocations


def test_get_source_no_observations(er_io):
    relocations = er_io.get_source_observations(
        source_ids=str(uuid.uuid4()),
        include_source_details=True,
    )
    assert relocations.empty


def test_get_subjectsource_observations(er_io):
    relocations = er_io.get_subjectsource_observations(
        subjectsource_ids=er_io.SUBJECTSOURCE_IDS,
        include_source_details=True,
    )
    assert isinstance(relocations, ecoscope.base.Relocations)
    assert "fixtime" in relocations
    assert "groupby_col" in relocations


def test_get_subjectsource_no_observations(er_io):
    relocations = er_io.get_subjectsource_observations(
        subjectsource_ids=str(uuid.uuid4()),
        include_source_details=True,
    )
    assert relocations.empty


def test_get_subjectgroup_observations(er_io):
    relocations = er_io.get_subjectgroup_observations(subject_group_name=er_io.GROUP_NAME)
    assert "groupby_col" in relocations
    assert len(relocations["extra__subject_id"].unique()) == 2


def test_get_events(er_events_io):
    events = er_events_io.get_events(event_type=["e00ce1f6-f9f1-48af-93c9-fb89ec493b8a"])
    assert not events.empty


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
    assert "time" in events


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
        since=pd.Timestamp("2017-01-01").isoformat(),
        until=pd.Timestamp("2017-04-01").isoformat(),
    )

    observations = er_io.get_patrol_observations(
        patrols,
        include_source_details=False,
        include_subject_details=False,
        include_subjectsource_details=False,
    )
    assert not observations.empty


def test_users(er_io):
    users = pd.DataFrame(er_io.get_users())
    assert not users.empty


def test_get_spatial_feature(er_io):
    spatial_feature = er_io.get_spatial_feature(spatial_feature_id="8868718f-0154-45bf-a74d-a66706ef958f")
    assert not spatial_feature.empty


def test_get_spatial_features_group(er_io):
    spatial_features = er_io.get_spatial_features_group(
        spatial_features_group_id="15698426-7e0f-41df-9bc3-495d87e2e097"
    )
    assert not spatial_features.empty


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

    new_client = ecoscope.io.EarthRangerIO(service_root=er_io.service_root, token_url=er_io.token_url, token=token)

    with pytest.raises(ERClientException):
        new_client.get_patrol_events(
            since=pd.Timestamp("2017-01-01").isoformat(),
            until=pd.Timestamp("2017-04-01").isoformat(),
        )
