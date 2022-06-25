import datetime
import uuid
from tempfile import TemporaryDirectory

import geopandas as gpd
import pandas as pd
import pytest
import pytz
from shapely.geometry import Point

import ecoscope

if not pytest.earthranger:
    pytest.skip(
        "Skipping tests because connection to EarthRanger is not available.",
        allow_module_level=True,
    )


GROUP_NAME = "Rhinos"  # Fatu, Najin
UNKNOWN_UUID = ["b03c362d-f41b-48ae-8584-3cbb6b016950"]
SUBJECT_IDS = [
    "216b033d-c4ad-444a-9f41-baa6c97cde7d",  # Fatu
    "aca64374-a102-4ef5-9b58-60fd0bf64a61",  # Najin
]
SOURCE_IDS = [
    "d8f92f6e-1121-4833-b11c-6fc1ca334ff0",  # Fatu
    "f46b6e92-a09d-41dd-bc42-8244870189fd",  # Najin
]
SUBJECTSOURCE_IDS = [
    "bce94e49-37b0-4fd0-a302-21a8daa245ff",  # Fatu
]
OBSERVATIONS = [
    {
        "recorded_at": datetime.datetime.utcnow(),
        "geometry": Point(0, 0),
        "source": SOURCE_IDS[0],
    },
    {
        "recorded_at": datetime.datetime.utcnow(),
        "geometry": Point(0, 0),
        "source": SOURCE_IDS[0],
    },
    {
        "recorded_at": datetime.datetime.utcnow(),
        "geometry": Point(1, 1.0),
        "source": SOURCE_IDS[1],
    },
]


def test_get_subject_observations(earthranger_io):
    relocations = earthranger_io.get_subject_observations(
        subject_ids=SUBJECT_IDS,
        include_subject_details=True,
        include_source_details=True,
        include_subjectsource_details=True,
    )
    assert isinstance(relocations, ecoscope.base.Relocations)
    assert "groupby_col" in relocations
    assert "fixtime" in relocations
    assert "extra__source" in relocations


def test_get_subject_no_observations(earthranger_io):
    with pytest.raises(ecoscope.contrib.dasclient.DasClientNotFound):
        earthranger_io.get_subject_observations(
            subject_ids=UNKNOWN_UUID,
            include_subject_details=True,
            include_source_details=True,
            include_subjectsource_details=True,
        )


def test_get_source_observations(earthranger_io):
    relocations = earthranger_io.get_source_observations(
        source_ids=SOURCE_IDS,
        include_source_details=True,
    )
    assert isinstance(relocations, ecoscope.base.Relocations)
    assert "fixtime" in relocations
    assert "groupby_col" in relocations


def test_get_source_no_observations(earthranger_io):
    relocations = earthranger_io.get_source_observations(
        source_ids=UNKNOWN_UUID,
        include_source_details=True,
    )
    assert relocations.empty


def test_get_subjectsource_observations(earthranger_io):
    relocations = earthranger_io.get_subjectsource_observations(
        subjectsource_ids=SUBJECTSOURCE_IDS,
        include_source_details=True,
    )
    assert isinstance(relocations, ecoscope.base.Relocations)
    assert "fixtime" in relocations
    assert "groupby_col" in relocations


def test_get_subjectsource_no_observations(earthranger_io):
    relocations = earthranger_io.get_subjectsource_observations(
        subjectsource_ids=UNKNOWN_UUID,
        include_source_details=True,
    )
    assert relocations.empty


def test_get_subjectsource_observations_with_pagesize_one(earthranger_io):
    relocations = earthranger_io.get_subjectsource_observations(
        subjectsource_ids=SUBJECTSOURCE_IDS,
        include_source_details=True,
        page_size=1,
    )
    assert isinstance(relocations, ecoscope.base.Relocations)
    assert relocations.shape[0] == 1


def test_get_subjectgroup_observations(earthranger_io):
    relocations = earthranger_io.get_subjectgroup_observations(group_name=GROUP_NAME)
    assert "groupby_col" in relocations


def test_get_events(earthranger_io):
    events = earthranger_io.get_events(page_size=1000)
    assert len(events) <= 1000


@pytest.mark.filterwarnings("ignore:All-NaN slice encountered:RuntimeWarning")
@pytest.mark.filterwarnings("ignore:Mean of empty slice:RuntimeWarning")
def test_collar_voltage(earthranger_io):
    start_time = pytz.utc.localize(datetime.datetime.now() - datetime.timedelta(days=31))
    observations = earthranger_io.get_subjectgroup_observations(
        group_name=GROUP_NAME,
        include_subject_details=True,
        include_subjectsource_details=True,
        include_details="true",
    )

    with TemporaryDirectory() as output_folder:
        ecoscope.plotting.plot.plot_collar_voltage(observations, start_time=start_time, output_folder=output_folder)


def test_das_client_method(earthranger_io):
    earthranger_io.pulse()
    earthranger_io.get_me()


def test_get_patrols(earthranger_io):
    patrols = earthranger_io.get_patrols()
    assert len(patrols) > 0


def test_post_observations(earthranger_io):
    observations = gpd.GeoDataFrame.from_dict(OBSERVATIONS)
    response = earthranger_io.post_observations(observations)
    assert len(response) == 3
    assert "location" in response
    assert "recorded_at" in response


def test_subjectsource(earthranger_io):
    response = earthranger_io.post_subjectsource(
        subject_id=SUBJECT_IDS[0],
        source_id=SOURCE_IDS[0],
        lower_bound_assignend_range=datetime.datetime.utcnow(),
        upper_bound_assigned_range=datetime.datetime.utcnow() + datetime.timedelta(days=30),
    )
    assert response.shape[0] == 1
    assert "216b033d-c4ad-444a-9f41-baa6c97cde7d" == response.subject[0]


def test_post_events(earthranger_io):
    events = [
        {
            "id": str(uuid.uuid4()),
            "title": "Accident",
            "event_type": "accident_rep",
            "time": pd.Timestamp.utcnow(),
            "location": {"longitude": "38.033294677734375", "latitude": "-2.9553841592697982"},
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
            "location": {"longitude": "38.4906005859375", "latitude": "-3.0321834919139206"},
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
    results = earthranger_io.post_event(events)
    results["time"] = pd.to_datetime(results["time"], utc=True)

    expected = pd.DataFrame(events)
    results = results[expected.columns]
    pd.testing.assert_frame_equal(results, expected)


def test_patch_event(earthranger_io):
    event = [
        {
            "id": str(uuid.uuid4()),
            "title": "Arrest",
            "event_type": "arrest_rep",
            "time": pd.Timestamp.utcnow(),
            "location": {"longitude": 38.11809539794921, "latitude": -3.4017015747197306},
            "priority": 200,
            "state": "new",
            "event_details": {
                "arrestrep_dateofbirth": "1985-01-1T13:00:00.000Z",
                "arrestrep_nationality": "other",
                "arrestrep_timeofarrest": datetime.datetime.utcnow().isoformat(),
                "arrestrep_reaonforarrest": "firearm",
                "arrestrep_arrestingranger": "Ranger Siera",
            },
            "is_collection": False,
            "icon_id": "arrest_rep",
        }
    ]
    earthranger_io.post_event(event)
    event_id = event[0]["id"]

    updated_event = pd.DataFrame(
        [
            {
                "priority": 300,
                "state": "active",
                "location": {"longitude": "38.4576416015625", "latitude": "-4.135503657998179"},
            }
        ]
    )

    result = earthranger_io.patch_event(event_id=event_id, events=updated_event)
    result = result[["priority", "state", "location"]]
    pd.testing.assert_frame_equal(result, updated_event)


def test_get_observation_for_patrol(earthranger_io):
    patrols = earthranger_io.get_patrols()
    observations = earthranger_io.get_observations_for_patrols(
        patrols,
        include_source_details=False,
        include_subject_details=False,
        include_subjectsource_details=False,
    )
    assert not observations.empty


def test_users(earthranger_io):
    users = earthranger_io.get_users()
    assert not users.empty
