import os

import pandas as pd
import pytest
import pytest_asyncio
import ecoscope
from erclient import ERClientException

pytestmark = pytest.mark.io


@pytest_asyncio.fixture
async def er_io_async():
    ER_SERVER = "https://mep-dev.pamdas.org"
    ER_USERNAME = os.getenv("ER_USERNAME")
    ER_PASSWORD = os.getenv("ER_PASSWORD")
    er_io = await ecoscope.io.AsyncEarthRangerIO.create(server=ER_SERVER, username=ER_USERNAME, password=ER_PASSWORD)

    return er_io


@pytest.fixture
def get_events_fields():
    return [
        "location",
        "time",
        "end_time",
        "message",
        "provenance",
        "event_type",
        "priority",
        "priority_label",
        "attributes",
        "comment",
        "title",
        "reported_by",
        "state",
        "is_contained_in",
        "sort_at",
        "patrol_segments",
        "geometry",
        "updated_at",
        "created_at",
        "icon_id",
        "serial_number",
        "event_details",
        "files",
        "related_subjects",
        "event_category",
        "url",
        "image_url",
        "geojson",
        "is_collection",
        "patrols",
    ]


@pytest.fixture
def get_source_fields():
    return [
        "id",
        "source_type",
        "manufacturer_id",
        "model_name",
        "additional",
        "provider",
        "content_type",
        "created_at",
        "updated_at",
        "url",
    ]


@pytest.fixture
def get_subjects_fields():
    return [
        "content_type",
        "id",
        "name",
        "subject_type",
        "subject_subtype",
        "common_name",
        "additional",
        "created_at",
        "updated_at",
        "is_active",
        "user",
        "tracks_available",
        "image_url",
        "last_position_status",
        "device_status_properties",
        "url",
        "last_position_date",
        "last_position",
        "region",
        "country",
        "sex",
        "hex",
    ]


@pytest.fixture
def get_patrols_fields():
    return [
        "id",
        "priority",
        "state",
        "objective",
        "serial_number",
        "title",
        "files",
        "notes",
        "patrol_segments",
        "updates",
    ]


@pytest.fixture
def get_patrol_observations_fields():
    return [
        "extra__id",
        "extra__location",
        "extra__recorded_at",
        "extra__created_at",
        "extra__exclusion_flags",
        "extra__source",
        "extra__subject_id",
        "geometry",
        "groupby_col",
        "fixtime",
        "junk_status",
    ]


@pytest.fixture
def get_patrol_details_fields():
    return [
        "patrol_id",
        "patrol_title",
        "patrol_serial_number",
        "patrol_start_time",
        "patrol_end_time",
        "patrol_type",
        "patrol_type__value",
        "patrol_type__display",
    ]


@pytest.fixture
def get_subjectsources_fields():
    return ["id", "assigned_range", "source", "subject", "additional", "location"]


@pytest.mark.asyncio
async def test_get_events_by_type(er_io_async, get_events_fields):
    # e00ce1f6-f9f1-48af-93c9-fb89ec493b8a == mepdev_distance_count
    events = await er_io_async.get_events_dataframe(event_type="e00ce1f6-f9f1-48af-93c9-fb89ec493b8a")
    assert not events.empty
    assert set(events.columns) == set(get_events_fields)
    assert type(events["time"] == pd.Timestamp)
    assert events["event_type"][0] == "mepdev_distance_count"


@pytest.mark.asyncio
async def test_get_events(er_io_async, get_events_fields):
    events = await er_io_async.get_events_dataframe(event_ids=["34ecf597-0ecc-4ac3-bec5-e9de801f0063"])
    assert len(events) == 1
    assert set(events.columns) == set(get_events_fields)
    assert type(events["time"] == pd.Timestamp)


@pytest.mark.asyncio
async def test_get_sources(er_io_async, get_source_fields):
    sources = await er_io_async.get_sources_dataframe()
    assert not sources.empty
    assert set(sources.columns) == set(get_source_fields)
    assert type(sources["updated_at"] == pd.Timestamp)


@pytest.mark.asyncio
async def test_get_subjects(er_io_async, get_subjects_fields):
    subjects = await er_io_async.get_subjects_dataframe()
    assert not subjects.empty
    assert set(subjects.columns) == set(get_subjects_fields)
    assert type(subjects["updated_at"] == pd.Timestamp)


@pytest.mark.asyncio
async def test_get_subjects_by_group_name(er_io_async, get_subjects_fields):
    subjects = await er_io_async.get_subjects_dataframe(subject_group_name="Elephants")
    assert not subjects.empty
    assert set(subjects.columns) == set(get_subjects_fields)
    assert type(subjects["updated_at"] == pd.Timestamp)


@pytest.mark.asyncio
async def test_get_subjectsources_by_subject(er_io_async, get_subjectsources_fields):
    subjectsources = await er_io_async.get_subjectsources_dataframe(subjects="5d600698-bcfa-401c-a617-e58e961a8038")
    assert not subjectsources.empty
    assert set(subjectsources.columns) == set(get_subjectsources_fields)


@pytest.mark.asyncio
async def test_get_subjectsources_by_source(er_io_async, get_subjectsources_fields):
    subjectsources = await er_io_async.get_subjectsources_dataframe(subjects="c9dbf311-144c-4e74-9b56-30771225dcd1")
    assert not subjectsources.empty
    assert set(subjectsources.columns) == set(get_subjectsources_fields)


@pytest.mark.asyncio
async def test_get_patrols(er_io_async, get_patrols_fields):
    patrols = await er_io_async.get_patrols_dataframe()
    assert not patrols.empty
    assert set(patrols.columns) == set(get_patrols_fields)


@pytest.mark.asyncio
async def test_get_patrol_observations(er_io_async, get_patrol_observations_fields):
    observations = await er_io_async.get_patrol_observations_with_patrol_filter(
        since=pd.Timestamp("2017-01-01").isoformat(),
        until=pd.Timestamp("2017-04-01").isoformat(),
    )
    assert not observations.empty
    assert set(observations.columns) == set(get_patrol_observations_fields)
    assert type(observations["fixtime"] == pd.Timestamp)


@pytest.mark.asyncio
async def test_get_patrol_observations_with_patrol_details(
    er_io_async, get_patrol_observations_fields, get_patrol_details_fields
):
    observations = await er_io_async.get_patrol_observations_with_patrol_filter(
        since=pd.Timestamp("2017-01-01").isoformat(),
        until=pd.Timestamp("2017-04-01").isoformat(),
        include_patrol_details=True,
    )
    assert not observations.empty
    assert set(observations.columns) == set(get_patrol_observations_fields).union(get_patrol_details_fields)
    assert type(observations["fixtime"] == pd.Timestamp)
    pd.testing.assert_series_equal(observations["patrol_id"], observations["groupby_col"], check_names=False)


@pytest.mark.asyncio
async def test_display_map(er_io_async):
    await er_io_async.load_display_map()
    assert er_io_async.event_type_display_values is not None
    assert len(er_io_async.event_type_display_values) > 0
    assert await er_io_async.get_event_type_display_name(event_type="fence_rep") == "Fence"
    assert (
        await er_io_async.get_event_type_display_name(event_type="shot_rep", event_property="shotrep_timeofshot")
        == "Time when shot was heard"
    )


@pytest.mark.asyncio
async def test_existing_token(er_io_async):
    await er_io_async.login()
    new_client = ecoscope.io.AsyncEarthRangerIO(
        service_root=er_io_async.service_root,
        token_url=er_io_async.token_url,
        token=er_io_async.auth.get("access_token"),
    )

    sources = await new_client.get_sources_dataframe()
    assert not sources.empty


@pytest.mark.asyncio
async def test_existing_token_expired(er_io_async):
    await er_io_async.login()
    token = er_io_async.auth.get("access_token")
    await er_io_async.refresh_token()

    new_client = ecoscope.io.AsyncEarthRangerIO(
        service_root=er_io_async.service_root, token_url=er_io_async.token_url, token=token
    )

    with pytest.raises(ERClientException):
        await new_client.get_sources_dataframe()


@pytest.mark.asyncio
async def test_get_me(er_io_async):
    me = await er_io_async.get_me()
    assert me.get("username")
