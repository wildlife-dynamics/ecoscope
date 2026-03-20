import os
from datetime import datetime, timezone
from unittest.mock import MagicMock, call, patch

import geopandas as gpd  # type: ignore[import-untyped]
import pandas as pd
import pytest
from wt_task import task

from ecoscope.platform.connections import EarthRangerConnection
from ecoscope.platform.tasks.filter._filter import UTC_TIMEZONEINFO, TimeRange
from ecoscope.platform.tasks.io import (
    get_analysis_field_from_event_details,
    get_analysis_field_label_from_event_details,
    get_analysis_field_unit_from_event_details,
    get_category_field_from_event_details,
    get_category_field_label_from_event_details,
    get_choices_from_v2_event_type,
    get_event_type_display_names_from_events,
    get_event_type_from_event_details,
    get_events,
    get_patrol_events,
    get_patrol_events_from_combined_params,
    get_patrol_observations,
    get_patrol_observations_from_combined_params,
    get_patrol_observations_from_patrols_df,
    get_patrol_observations_from_patrols_df_and_combined_params,
    get_patrols,
    get_patrols_from_combined_params,
    get_spatial_features_group,
    get_subjectgroup_observations,
    set_event_details_params,
    set_patrols_and_patrol_events_params,
    unpack_events_from_patrols_df,
    unpack_events_from_patrols_df_and_combined_params,
)
from ecoscope.platform.tasks.io._earthranger import (
    _EXCLUSION_FILTER_TO_INT,
    CombinedPatrolAndEventsParams,
    _make_warehouse_client_from_env,
)

pytestmark = pytest.mark.io


@pytest.fixture(scope="session")
def client():
    return EarthRangerConnection(
        server=os.environ["EARTHRANGER_SERVER"],
        username=os.environ["EARTHRANGER_USERNAME"],
        password=os.environ["EARTHRANGER_PASSWORD"],
        tcp_limit="5",
        sub_page_size="4000",
    ).get_client()


@pytest.fixture
def named_mock_env():
    return {
        "ECOSCOPE_WORKFLOWS__CONNECTIONS__EARTHRANGER__MEP_DEV__SERVER": (os.environ["EARTHRANGER_SERVER"]),
        "ECOSCOPE_WORKFLOWS__CONNECTIONS__EARTHRANGER__MEP_DEV__USERNAME": os.environ["EARTHRANGER_USERNAME"],
        "ECOSCOPE_WORKFLOWS__CONNECTIONS__EARTHRANGER__MEP_DEV__PASSWORD": os.environ["EARTHRANGER_PASSWORD"],
        "ECOSCOPE_WORKFLOWS__CONNECTIONS__EARTHRANGER__MEP_DEV__TCP_LIMIT": "5",
        "ECOSCOPE_WORKFLOWS__CONNECTIONS__EARTHRANGER__MEP_DEV__SUB_PAGE_SIZE": "4000",
    }


@pytest.fixture
def mock_empty_client():
    mock = MagicMock()
    mock.get_patrols.return_value = pd.DataFrame()
    mock.get_subjectgroup_observations.return_value = pd.DataFrame()
    mock.get_patrol_observations_with_patrol_filter.return_value = pd.DataFrame()
    mock.get_patrol_events.return_value = pd.DataFrame()
    mock.get_events.return_value = pd.DataFrame()
    mock.get_event_types.return_value = pd.DataFrame({"id": ["AAA123"], "value": ["an_event"]})
    return mock


def test_get_subject_group_observations(client):
    result = get_subjectgroup_observations(
        client=client,
        subject_group_name="Ecoscope",
        time_range=TimeRange(
            since=datetime.strptime("2017-01-01", "%Y-%m-%d").replace(tzinfo=timezone.utc),
            until=datetime.strptime("2017-03-31", "%Y-%m-%d").replace(tzinfo=timezone.utc),
            timezone=UTC_TIMEZONEINFO,
        ),
    )

    assert len(result) > 0
    assert "geometry" in result
    assert "groupby_col" in result
    assert "fixtime" in result
    assert "junk_status" in result


def test_get_patrol_observations(named_mock_env):
    with patch.dict(os.environ, named_mock_env):
        result = (
            task(get_patrol_observations)
            .validate()
            .call(
                client="MEP_DEV",
                time_range={
                    "since": "2015-01-01T00:00:00Z",
                    "until": "2015-03-01T23:59:59Z",
                    "timezone": {
                        "label": "UTC",
                        "tzCode": "UTC",
                        "name": "UTC",
                        "utc": "+00:00",
                    },
                },
                patrol_types=["    ecoscope_patrol           "],  # whitespaces are intentional to test stripping
                status=None,
                include_patrol_details=True,
            )
        )

        assert len(result) > 0
        assert "geometry" in result
        assert "groupby_col" in result
        assert "fixtime" in result
        assert "junk_status" in result


def test_get_patrol_observations_with_whitespace_in_patrol_types_without_pydantic_validations(
    client,
):
    with pytest.raises(
        ValueError,
        match="Failed to find IDs for values",
    ):
        get_patrol_observations(
            client=client,
            time_range=TimeRange(
                since=datetime.strptime("2015-01-01", "%Y-%m-%d").replace(tzinfo=timezone.utc),
                until=datetime.strptime("2015-03-01", "%Y-%m-%d").replace(tzinfo=timezone.utc),
                timezone=UTC_TIMEZONEINFO,
            ),
            patrol_types=["    ecoscope_patrol           "],
            # the whitespaces here will NOT get stripped because we're not using
            # Pydantic validation. This should cause ERClient to signal an exception.
            status=None,
            include_patrol_details=True,
            raise_on_empty=True,
        )


def test_get_patrol_events(client):
    result = get_patrol_events(
        client=client,
        time_range=TimeRange(
            since=datetime.strptime("2015-01-01", "%Y-%m-%d").replace(tzinfo=timezone.utc),
            until=datetime.strptime("2015-03-01", "%Y-%m-%d").replace(tzinfo=timezone.utc),
            timezone=UTC_TIMEZONEINFO,
        ),
        patrol_types="ecoscope_patrol",
        event_types=[],
        status=None,
    )

    assert len(result) > 0
    assert "id" in result
    assert "event_type" in result
    assert "geometry" in result


def test_get_events(client):
    result = get_events(
        client=client,
        time_range=TimeRange(
            since=datetime.strptime("2015-01-01", "%Y-%m-%d").replace(tzinfo=timezone.utc),
            until=datetime.strptime("2015-12-31", "%Y-%m-%d").replace(tzinfo=timezone.utc),
            timezone=UTC_TIMEZONEINFO,
        ),
        event_types=[
            "hwc_rep",
            "bird_sighting_rep",
            "wildlife_sighting_rep",
            "poacher_camp_rep",
            "fire_rep",
            "injured_animal_rep",
        ],
        event_columns=["id", "time", "event_type", "geometry"],
    )

    assert len(result) > 0
    assert "id" in result
    assert "time" in result
    assert "event_type" in result
    assert "geometry" in result


def test_get_events_with_details(client):
    result = get_events(
        client=client,
        time_range=TimeRange(
            since=datetime.strptime("2015-01-01", "%Y-%m-%d").replace(tzinfo=timezone.utc),
            until=datetime.strptime("2015-12-31", "%Y-%m-%d").replace(tzinfo=timezone.utc),
            timezone=UTC_TIMEZONEINFO,
        ),
        include_details=True,
        include_updates=True,
        include_related_events=True,
        event_types=[
            "hwc_rep",
            "bird_sighting_rep",
            "wildlife_sighting_rep",
            "poacher_camp_rep",
            "fire_rep",
            "injured_animal_rep",
        ],
        event_columns=[
            "id",
            "time",
            "event_type",
            "geometry",
            "event_details",
            "updates",
            "is_linked_to",
        ],
    )

    assert len(result) > 0
    assert "id" in result
    assert "time" in result
    assert "event_type" in result
    assert "geometry" in result
    assert "event_details" in result
    assert "updates" in result
    assert "is_linked_to" in result


def test_get_events_with_event_type_whitespace(named_mock_env):
    with patch.dict(os.environ, named_mock_env):
        result = (
            task(get_events)
            .validate()
            .call(
                client="MEP_DEV",
                time_range={
                    "since": "2015-01-01T00:00:00Z",
                    "until": "2015-12-31T23:59:59Z",
                    "timezone": {
                        "label": "UTC",
                        "tzCode": "UTC",
                        "name": "UTC",
                        "utc": "+00:00",
                    },
                },
                event_types=[
                    "         hwc_rep    ",  # whitespaces are intentional to test stripping
                    "   bird_sighting_rep           ",  # whitespaces are intentional to test stripping
                    "      wildlife_sighting_rep        ",  # whitespaces are intentional to test stripping
                    "  poacher_camp_rep    ",  # whitespaces are intentional to test stripping
                    "    fire_rep   ",  # whitespaces are intentional to test stripping
                    "     injured_animal_rep   ",  # whitespaces are intentional to test stripping
                ],
                event_columns=["id", "time", "event_type", "geometry"],
            )
        )

        assert len(result) > 0
        assert "id" in result


def test_get_events_bad_event_type(client):
    result = get_events(
        client=client,
        time_range=TimeRange(
            since=datetime.strptime("2015-01-01", "%Y-%m-%d").replace(tzinfo=timezone.utc),
            until=datetime.strptime("2015-12-31", "%Y-%m-%d").replace(tzinfo=timezone.utc),
            timezone=UTC_TIMEZONEINFO,
        ),
        event_types=[
            "not a real type",
        ],
        raise_on_empty=False,
        event_columns=["id", "time", "event_type", "geometry"],
    )

    assert result.empty


def test_bad_token_fails():
    with pytest.raises(Exception, match="Authorization token is invalid or expired."):
        EarthRangerConnection(
            server=os.environ["EARTHRANGER_SERVER"],
            token="abc123",
            tcp_limit="5",
            sub_page_size="4000",
        ).get_client()


def test_get_patrol_observations_all_types(client):
    result = get_patrol_observations(
        client=client,
        time_range=TimeRange(
            since=datetime.strptime("2024-01-01", "%Y-%m-%d").replace(tzinfo=timezone.utc),
            until=datetime.strptime("2024-12-31", "%Y-%m-%d").replace(tzinfo=timezone.utc),
            timezone=UTC_TIMEZONEINFO,
        ),
        patrol_types=[],
        status=None,
        include_patrol_details=True,
    )
    assert "patrol_type" in result
    assert len(result["patrol_type"].unique()) > 1


def test_get_events_parity(client):
    """
    The intent of this test is to demonstrate that new args added to the signature of the get_events task
    do not alter the return value of the core library function in any way.
    """
    original_event_args = {
        "since": datetime.strptime("2015-01-01", "%Y-%m-%d").replace(tzinfo=timezone.utc).isoformat(),
        "until": datetime.strptime("2015-12-31", "%Y-%m-%d").replace(tzinfo=timezone.utc).isoformat(),
        "event_type_ids": ["9477c3e3-cf46-4f2e-9bdd-05f91b2201ba"],
        "drop_null_geometry": False,
    }

    # The new args added by #1058
    new_event_args = {
        "include_details": False,
        "include_updates": False,
        "include_related_events": False,
    }

    get_events_no_extras = client.get_events(**original_event_args)
    get_events_with_extras = client.get_events(**(original_event_args | new_event_args))

    pd.testing.assert_frame_equal(get_events_no_extras, get_events_with_extras)


def test_get_events_all_types(client):
    result = get_events(
        client=client,
        time_range=TimeRange(
            since=datetime.strptime("2015-01-01", "%Y-%m-%d").replace(tzinfo=timezone.utc),
            until=datetime.strptime("2015-12-31", "%Y-%m-%d").replace(tzinfo=timezone.utc),
            timezone=UTC_TIMEZONEINFO,
        ),
        event_types=[],
        event_columns=["id", "time", "event_type", "geometry"],
    )
    assert "event_type" in result
    assert len(result["event_type"].unique()) > 1


def test_get_patrol_events_all_types(client):
    result = get_patrol_events(
        client=client,
        time_range=TimeRange(
            since=datetime.strptime("2024-01-01", "%Y-%m-%d").replace(tzinfo=timezone.utc),
            until=datetime.strptime("2024-12-31", "%Y-%m-%d").replace(tzinfo=timezone.utc),
            timezone=UTC_TIMEZONEINFO,
        ),
        patrol_types=[],
        event_types=[],
    )

    assert "patrol_type" in result
    assert len(result["patrol_type"].unique()) > 1
    assert len(result["event_type"].unique()) > 1


def test_get_patrol_events_with_event_type_filter(client):
    event_type_filter = ["hwc_rep", "fire_rep"]
    result = get_patrol_events(
        client=client,
        time_range=TimeRange(
            since=datetime.strptime("2017-01-01", "%Y-%m-%d").replace(tzinfo=timezone.utc),
            until=datetime.strptime("2017-04-01", "%Y-%m-%d").replace(tzinfo=timezone.utc),
            timezone=UTC_TIMEZONEINFO,
        ),
        patrol_types=[],
        event_types=event_type_filter,
    )
    assert not result.empty
    assert result["event_type"].unique().tolist() == event_type_filter


@pytest.mark.parametrize(
    "filter_value, expected_int",
    list(_EXCLUSION_FILTER_TO_INT.items()),
)
def test_get_subjectgroup_observations_exclusion_filter(mock_empty_client, filter_value, expected_int):
    get_subjectgroup_observations(
        client=mock_empty_client,
        subject_group_name="Ecoscope",
        time_range=TimeRange(
            since=datetime.strptime("2017-01-01", "%Y-%m-%d").replace(tzinfo=timezone.utc),
            until=datetime.strptime("2017-03-31", "%Y-%m-%d").replace(tzinfo=timezone.utc),
            timezone=UTC_TIMEZONEINFO,
        ),
        raise_on_empty=False,
        filter=filter_value,
    )
    mock_empty_client.get_subjectgroup_observations.assert_called_once()
    call_kwargs = mock_empty_client.get_subjectgroup_observations.call_args.kwargs
    assert call_kwargs["filter"] == expected_int


def test_get_subjectgroup_observations_empty_response(mock_empty_client):
    kwargs = {
        "client": mock_empty_client,
        "subject_group_name": "Ecoscope",
        "time_range": TimeRange(
            since=datetime.strptime("2017-01-01", "%Y-%m-%d").replace(tzinfo=timezone.utc),
            until=datetime.strptime("2017-03-31", "%Y-%m-%d").replace(tzinfo=timezone.utc),
            timezone=UTC_TIMEZONEINFO,
        ),
    }

    with pytest.raises(ValueError, match=r"No data returned from EarthRanger.*"):
        get_subjectgroup_observations(**kwargs)

    kwargs["raise_on_empty"] = False
    df = get_subjectgroup_observations(**kwargs)
    assert df.empty


def test_get_patrol_observations_empty_response(mock_empty_client):
    kwargs = {
        "client": mock_empty_client,
        "time_range": TimeRange(
            since=datetime.strptime("2017-01-01", "%Y-%m-%d").replace(tzinfo=timezone.utc),
            until=datetime.strptime("2017-03-31", "%Y-%m-%d").replace(tzinfo=timezone.utc),
            timezone=UTC_TIMEZONEINFO,
        ),
        "patrol_types": "ecoscope_patrol",
    }

    with pytest.raises(ValueError, match=r"No data returned from EarthRanger.*"):
        get_patrol_observations(**kwargs)

    kwargs["raise_on_empty"] = False
    df = get_patrol_observations(**kwargs)
    assert df.empty


def test_get_patrol_events_empty_response(mock_empty_client):
    kwargs = {
        "client": mock_empty_client,
        "time_range": TimeRange(
            since=datetime.strptime("2017-01-01", "%Y-%m-%d").replace(tzinfo=timezone.utc),
            until=datetime.strptime("2017-03-31", "%Y-%m-%d").replace(tzinfo=timezone.utc),
            timezone=UTC_TIMEZONEINFO,
        ),
        "patrol_types": "ecoscope_patrol",
        "event_types": [],
    }

    with pytest.raises(ValueError, match=r"No data returned from EarthRanger.*"):
        get_patrol_events(**kwargs)

    kwargs["raise_on_empty"] = False
    df = get_patrol_events(**kwargs)
    assert df.empty


def test_get_events_empty_response(mock_empty_client):
    kwargs = {
        "client": mock_empty_client,
        "time_range": TimeRange(
            since=datetime.strptime("2017-01-01", "%Y-%m-%d").replace(tzinfo=timezone.utc),
            until=datetime.strptime("2017-03-31", "%Y-%m-%d").replace(tzinfo=timezone.utc),
            timezone=UTC_TIMEZONEINFO,
        ),
        "event_types": ["an_event"],
        "event_columns": ["id", "time", "event_type", "geometry"],
    }

    with pytest.raises(ValueError, match=r"No data returned from EarthRanger.*"):
        get_events(**kwargs)

    kwargs["raise_on_empty"] = False
    df = get_events(**kwargs)
    assert df.empty


def test_get_patrols_empty_response(mock_empty_client):
    kwargs = {
        "client": mock_empty_client,
        "time_range": TimeRange(
            since=datetime.strptime("2017-01-01", "%Y-%m-%d").replace(tzinfo=timezone.utc),
            until=datetime.strptime("2017-03-31", "%Y-%m-%d").replace(tzinfo=timezone.utc),
            timezone=UTC_TIMEZONEINFO,
        ),
        "patrol_types": "ecoscope_patrol",
    }

    with pytest.raises(ValueError, match=r"No data returned from EarthRanger.*"):
        get_patrols(**kwargs)

    kwargs["raise_on_empty"] = False
    df = get_patrols(**kwargs)
    assert df.empty


def test_get_patrol_observations_from_patrols_df_empty_response(mock_empty_client):
    empty_patrols_df = pd.DataFrame()
    kwargs = {
        "client": mock_empty_client,
        "patrols_df": empty_patrols_df,
    }

    with pytest.raises(ValueError, match=r"No data returned from EarthRanger.*"):
        get_patrol_observations_from_patrols_df(**kwargs)

    kwargs["raise_on_empty"] = False
    df = get_patrol_observations_from_patrols_df(**kwargs)
    assert df.empty


def test_unpack_events_from_patrols_df_empty_response(mock_empty_client):
    empty_patrols_df = pd.DataFrame()
    kwargs = {
        "patrols_df": empty_patrols_df,
        "event_types": [],
        "time_range": TimeRange(
            since=datetime.strptime("2017-01-01", "%Y-%m-%d").replace(tzinfo=timezone.utc),
            until=datetime.strptime("2017-03-31", "%Y-%m-%d").replace(tzinfo=timezone.utc),
            timezone=UTC_TIMEZONEINFO,
        ),
    }

    with pytest.raises(ValueError, match=r"No event data in provided patrols_df"):
        unpack_events_from_patrols_df(**kwargs)

    kwargs["raise_on_empty"] = False
    df = unpack_events_from_patrols_df(**kwargs)
    assert df.empty


def test_patrol_events_combined(named_mock_env):
    patrol_obs_args = {
        "client": "MEP_DEV",
        "time_range": TimeRange(
            since=datetime.strptime("2024-01-01", "%Y-%m-%d").replace(tzinfo=timezone.utc),
            until=datetime.strptime("2024-12-31", "%Y-%m-%d").replace(tzinfo=timezone.utc),
            timezone=UTC_TIMEZONEINFO,
        ),
        "patrol_types": ["ecoscope_patrol"],
        "status": None,
        "include_patrol_details": True,
        "raise_on_empty": False,
        "sub_page_size": 100,
    }
    patrol_events_args = {
        "client": "MEP_DEV",
        "time_range": TimeRange(
            since=datetime.strptime("2024-01-01", "%Y-%m-%d").replace(tzinfo=timezone.utc),
            until=datetime.strptime("2024-12-31", "%Y-%m-%d").replace(tzinfo=timezone.utc),
            timezone=UTC_TIMEZONEINFO,
        ),
        "patrol_types": ["ecoscope_patrol"],
        "event_types": [],
        "status": None,
        "raise_on_empty": False,
        "include_null_geometry": True,
        "sub_page_size": 100,
    }
    combined_args = patrol_obs_args | patrol_events_args

    expected_patrol_obs_call_args = {
        "since": patrol_obs_args["time_range"].since.isoformat(),
        "until": patrol_obs_args["time_range"].until.isoformat(),
        "patrol_type_value": patrol_obs_args["patrol_types"],
        "status": [
            "done"  # Since status is None in the task args we expect the default value here
        ],
        "include_patrol_details": patrol_obs_args["include_patrol_details"],
        "sub_page_size": 100,
        "patrols_overlap_daterange": True,
    }
    expected_patrol_events_call_args = {
        "since": patrol_events_args["time_range"].since.isoformat(),
        "until": patrol_events_args["time_range"].until.isoformat(),
        "patrol_type_value": patrol_events_args["patrol_types"],
        "event_type": patrol_events_args["event_types"],
        "status": [
            "done"  # Since status is None in the task args we expect the default value here
        ],
        # We expect this to be inverted since this is checked against the core lib
        "drop_null_geometry": not patrol_events_args["include_null_geometry"],
        "sub_page_size": 100,
        "patrols_overlap_daterange": True,
    }

    with patch.dict(os.environ, named_mock_env):
        mock_patrol_obs = MagicMock(return_value=pd.DataFrame())
        mock_patrol_events = MagicMock(return_value=pd.DataFrame())
        with patch(
            "ecoscope.io.EarthRangerIO.get_patrol_observations_with_patrol_filter",
            mock_patrol_obs,
        ):
            with patch("ecoscope.io.EarthRangerIO.get_patrol_events", mock_patrol_events):
                task(get_patrol_observations).validate().call(**patrol_obs_args)
                task(get_patrol_events).validate().call(**patrol_events_args)

                combined_params = set_patrols_and_patrol_events_params(**combined_args)
                get_patrol_events_from_combined_params(combined_params)
                get_patrol_observations_from_combined_params(combined_params)

                # Check that the underlying IO calls were made with identical args
                mock_patrol_obs.assert_has_calls(
                    [
                        call(**expected_patrol_obs_call_args),
                        call(**expected_patrol_obs_call_args),
                    ]
                )
                mock_patrol_events.assert_has_calls(
                    [
                        call(**expected_patrol_events_call_args),
                        call(**expected_patrol_events_call_args),
                    ]
                )


def test_get_patrol_observations_combined_parity(named_mock_env):
    with patch.dict(os.environ, named_mock_env):
        args = {
            "client": "MEP_DEV",
            "time_range": TimeRange(
                since=datetime.strptime("2015-01-01", "%Y-%m-%d").replace(tzinfo=timezone.utc),
                until=datetime.strptime("2015-03-01", "%Y-%m-%d").replace(tzinfo=timezone.utc),
                timezone=UTC_TIMEZONEINFO,
            ),
            "patrol_types": ["ecoscope_patrol"],
            "status": None,
            "sub_page_size": 100,
        }

        result = task(get_patrol_observations).validate().call(**args)
        assert len(result) > 0
        assert "geometry" in result
        assert "groupby_col" in result
        assert "fixtime" in result
        assert "junk_status" in result

        # event_types is not an arg in get_patrol_observations, but it is required for
        # CombinedPatrolAndEventsParams so explicitly add here
        combined = CombinedPatrolAndEventsParams(**args, event_types=[])
        pd.testing.assert_frame_equal(get_patrol_observations_from_combined_params(combined), result)


def test_get_patrol_events_combined_parity(named_mock_env):
    with patch.dict(os.environ, named_mock_env):
        args = {
            "client": "MEP_DEV",
            "time_range": TimeRange(
                since=datetime.strptime("2015-01-01", "%Y-%m-%d").replace(tzinfo=timezone.utc),
                until=datetime.strptime("2015-03-01", "%Y-%m-%d").replace(tzinfo=timezone.utc),
                timezone=UTC_TIMEZONEINFO,
            ),
            "patrol_types": ["ecoscope_patrol"],
            "event_types": [],
            "status": None,
            "sub_page_size": 100,
        }

        result = task(get_patrol_events).validate().call(**args)
        assert len(result) > 0
        assert "id" in result
        assert "event_type" in result
        assert "geometry" in result

        combined = CombinedPatrolAndEventsParams(**args)
        pd.testing.assert_frame_equal(get_patrol_events_from_combined_params(combined), result)


def test_get_patrols_combined_parity(named_mock_env):
    with patch.dict(os.environ, named_mock_env):
        args = {
            "client": "MEP_DEV",
            "time_range": TimeRange(
                since=datetime.strptime("2015-01-01", "%Y-%m-%d").replace(tzinfo=timezone.utc),
                until=datetime.strptime("2015-03-01", "%Y-%m-%d").replace(tzinfo=timezone.utc),
                timezone=UTC_TIMEZONEINFO,
            ),
            "patrol_types": ["ecoscope_patrol"],
            "status": None,
            "sub_page_size": 100,
        }

        result = task(get_patrols).validate().call(**args)
        assert len(result) > 0
        assert "id" in result
        assert "state" in result
        assert "patrol_segments" in result
        assert "geometry" not in result

        # event_types is not an arg in get_patrols, but it is required for
        # CombinedPatrolAndEventsParams so explicitly add here
        combined = CombinedPatrolAndEventsParams(**args, event_types=[])
        pd.testing.assert_frame_equal(get_patrols_from_combined_params(combined), result)


def test_get_patrol_observations_from_patrols_df_combined_parity(client, named_mock_env):
    time_range = TimeRange(
        since=datetime.strptime("2015-01-01", "%Y-%m-%d").replace(tzinfo=timezone.utc),
        until=datetime.strptime("2015-03-01", "%Y-%m-%d").replace(tzinfo=timezone.utc),
        timezone=UTC_TIMEZONEINFO,
    )
    patrol_types = "ecoscope_patrol"
    status = None

    patrols_df = get_patrols(
        client=client,
        time_range=time_range,
        patrol_types=patrol_types,
        status=status,
    )
    with patch.dict(os.environ, named_mock_env):
        shared_args = {
            "client": "MEP_DEV",
            "include_patrol_details": True,
        }

        result = task(get_patrol_observations_from_patrols_df).validate().call(**shared_args, patrols_df=patrols_df)
        assert len(result) > 0
        assert "geometry" in result
        assert "groupby_col" in result
        assert "fixtime" in result
        assert "junk_status" in result

        combined = CombinedPatrolAndEventsParams(
            **shared_args,
            time_range=time_range,
            patrol_types=patrol_types,
            status=status,
            event_types=[],
        )
        pd.testing.assert_frame_equal(
            get_patrol_observations_from_patrols_df_and_combined_params(
                patrols_df=patrols_df, combined_params=combined
            ),
            result,
        )


def test_unpack_events_from_patrols_df_combined_parity(client, named_mock_env):
    time_range = TimeRange(
        since=datetime.strptime("2015-01-01", "%Y-%m-%d").replace(tzinfo=timezone.utc),
        until=datetime.strptime("2015-03-01", "%Y-%m-%d").replace(tzinfo=timezone.utc),
        timezone=UTC_TIMEZONEINFO,
    )
    patrol_types = "ecoscope_patrol"
    status = None

    patrols_df = get_patrols(
        client=client,
        time_range=time_range,
        patrol_types=patrol_types,
        status=status,
    )
    with patch.dict(os.environ, named_mock_env):
        shared_args = {
            "time_range": time_range,
            "event_types": [],
        }
        result = task(unpack_events_from_patrols_df).validate().call(**shared_args, patrols_df=patrols_df)
        assert len(result) > 0
        assert "id" in result
        assert "event_type" in result
        assert "geometry" in result

        combined = CombinedPatrolAndEventsParams(
            **shared_args,
            client=client,
            patrol_types=patrol_types,
            status=status,
        )
        pd.testing.assert_frame_equal(
            unpack_events_from_patrols_df_and_combined_params(patrols_df=patrols_df, combined_params=combined),
            result,
        )


def test_get_patrols(client):
    result = get_patrols(
        client=client,
        time_range=TimeRange(
            since=datetime.strptime("2015-01-01", "%Y-%m-%d").replace(tzinfo=timezone.utc),
            until=datetime.strptime("2015-03-01", "%Y-%m-%d").replace(tzinfo=timezone.utc),
            timezone=UTC_TIMEZONEINFO,
        ),
        patrol_types="ecoscope_patrol",
        status=None,
    )

    assert len(result) > 0
    assert "id" in result
    assert "state" in result
    assert "patrol_segments" in result
    assert "geometry" not in result


def test_get_patrol_observations_from_patrols_df(client):
    patrols = get_patrols(
        client=client,
        time_range=TimeRange(
            since=datetime.strptime("2015-01-01", "%Y-%m-%d").replace(tzinfo=timezone.utc),
            until=datetime.strptime("2015-03-01", "%Y-%m-%d").replace(tzinfo=timezone.utc),
            timezone=UTC_TIMEZONEINFO,
        ),
        patrol_types="ecoscope_patrol",
        status=None,
    )
    relocs = get_patrol_observations_from_patrols_df(
        client=client,
        patrols_df=patrols,
        include_patrol_details=True,
    )

    assert len(relocs) > 0
    assert "geometry" in relocs
    assert "groupby_col" in relocs
    assert "fixtime" in relocs
    assert "junk_status" in relocs


def test_unpack_events_from_patrols_df(client):
    time_range = TimeRange(
        since=datetime.strptime("2015-01-01", "%Y-%m-%d").replace(tzinfo=timezone.utc),
        until=datetime.strptime("2015-03-01", "%Y-%m-%d").replace(tzinfo=timezone.utc),
        timezone=UTC_TIMEZONEINFO,
    )
    patrols = get_patrols(
        client=client,
        time_range=time_range,
        patrol_types="ecoscope_patrol",
        status=None,
    )
    events = unpack_events_from_patrols_df(
        patrols_df=patrols,
        time_range=time_range,
        event_types=[],
    )

    assert len(events) > 0
    assert "id" in events
    assert "event_type" in events
    assert "geometry" in events


def test_get_events_empty_event_type_selection(client):
    result = get_events(
        client=client,
        time_range=TimeRange(
            since=datetime.strptime("2015-01-01", "%Y-%m-%d").replace(tzinfo=timezone.utc),
            until=datetime.strptime("2015-12-31", "%Y-%m-%d").replace(tzinfo=timezone.utc),
            timezone=UTC_TIMEZONEINFO,
        ),
        event_types=[],
        event_columns=["id", "time", "event_type", "geometry"],
    )

    assert len(result) > 0
    assert "id" in result
    assert "time" in result
    assert "event_type" in result
    assert "geometry" in result
    assert len(result["event_type"].unique()) > 1


def test_get_event_type_display_names_from_events(client):
    events_gdf = get_events(
        client=client,
        time_range=TimeRange(
            since=datetime.strptime("2025-01-01", "%Y-%m-%d").replace(tzinfo=timezone.utc),
            until=datetime.strptime("2025-12-31", "%Y-%m-%d").replace(tzinfo=timezone.utc),
            timezone=UTC_TIMEZONEINFO,
        ),
        event_types=[],
        event_columns=["id", "time", "event_type", "geometry"],
    )

    with_display_types = get_event_type_display_names_from_events(
        client=client,
        events_gdf=events_gdf,
        append_category_names="duplicates",
    )
    assert "event_type_display" in with_display_types.columns


def test_get_patrol_events_with_display_values(client):
    result = get_patrol_events(
        client=client,
        time_range=TimeRange(
            since=datetime.strptime("2015-01-01", "%Y-%m-%d").replace(tzinfo=timezone.utc),
            until=datetime.strptime("2015-03-01", "%Y-%m-%d").replace(tzinfo=timezone.utc),
            timezone=UTC_TIMEZONEINFO,
        ),
        patrol_types="ecoscope_patrol",
        event_types=[],
        status=None,
        include_display_values=True,
    )

    assert len(result) > 0
    assert "id" in result
    assert "event_type" in result
    assert "event_type_display" in result
    assert "geometry" in result


def test_get_patrol_events_with_display_values_empty(client):
    result = get_patrol_events(
        client=client,
        time_range=TimeRange(
            since=datetime.strptime("1985-01-01", "%Y-%m-%d").replace(tzinfo=timezone.utc),
            until=datetime.strptime("1985-03-01", "%Y-%m-%d").replace(tzinfo=timezone.utc),
            timezone=UTC_TIMEZONEINFO,
        ),
        patrol_types="ecoscope_patrol",
        event_types=[],
        status=None,
        raise_on_empty=False,
        include_display_values=True,
    )

    assert result.empty


def test_get_events_with_display_values(client):
    result = get_events(
        client=client,
        time_range=TimeRange(
            since=datetime.strptime("2015-01-01", "%Y-%m-%d").replace(tzinfo=timezone.utc),
            until=datetime.strptime("2015-12-31", "%Y-%m-%d").replace(tzinfo=timezone.utc),
            timezone=UTC_TIMEZONEINFO,
        ),
        event_types=[
            "hwc_rep",
            "bird_sighting_rep",
            "wildlife_sighting_rep",
            "poacher_camp_rep",
            "fire_rep",
            "injured_animal_rep",
        ],
        event_columns=["id", "time", "event_type", "geometry"],
        include_display_values=True,
    )

    assert len(result) > 0
    assert "id" in result
    assert "time" in result
    assert "event_type" in result
    assert "event_type_display" in result
    assert "geometry" in result


def test_get_events_with_display_values_empty(client):
    result = get_events(
        client=client,
        time_range=TimeRange(
            since=datetime.strptime("1985-01-01", "%Y-%m-%d").replace(tzinfo=timezone.utc),
            until=datetime.strptime("1985-12-31", "%Y-%m-%d").replace(tzinfo=timezone.utc),
            timezone=UTC_TIMEZONEINFO,
        ),
        event_types=[],
        event_columns=["id", "time", "event_type", "geometry"],
        raise_on_empty=False,
        include_display_values=True,
    )

    assert result.empty


def test_get_choices_from_v2_event_type(client):
    result = get_choices_from_v2_event_type(
        client=client,
        event_type="elephant_sigthing_test",
        choice_field="herd_type",
    )

    assert {
        "bull_only": "Bull Only",
        "female_only": "Female Only",
        "mix": "Mix",
    } == result


def test_get_choices_from_v2_event_type_not_found(client):
    result = get_choices_from_v2_event_type(
        client=client,
        event_type="accident_rep",
        choice_field="herd_type",
    )

    assert {} == result


def test_get_choices_from_v2_event_type_choice_is_none(client):
    result = get_choices_from_v2_event_type(
        client=client,
        event_type="accident_rep",
        choice_field=None,
    )

    assert {} == result


def test_event_details_params_emitters(client):
    input_time_range = TimeRange(
        since=datetime.strptime("2017-01-01", "%Y-%m-%d").replace(tzinfo=timezone.utc),
        until=datetime.strptime("2017-03-31", "%Y-%m-%d").replace(tzinfo=timezone.utc),
        timezone=UTC_TIMEZONEINFO,
    )
    input_event_type = "an_event_type"
    input_event_columns = ["serial_number"]
    input_analysis_field = "a_numeric_field"
    input_analysis_field_label = "Numeric Field Label"
    input_analysis_field_unit = "Fields"
    input_category_field = "an_category_field"
    input_category_field_label = "An Category Field Label"

    params = set_event_details_params(
        client=client,
        time_range=input_time_range,
        event_type=input_event_type,
        event_columns=input_event_columns,
        analysis_field=input_analysis_field,
        analysis_field_label=input_analysis_field_label,
        analysis_field_unit=input_analysis_field_unit,
        category_field=input_category_field,
        category_field_label=input_category_field_label,
    )

    assert params.get_events_params() == {
        "client": client,
        "time_range": input_time_range,
        "event_types": [input_event_type],
        "event_columns": input_event_columns,
        "include_null_geometry": True,
        "raise_on_empty": True,
        "include_details": True,
        "include_updates": False,
        "include_related_events": False,
        "include_display_values": False,
    }

    assert get_event_type_from_event_details(params) == input_event_type
    assert get_analysis_field_from_event_details(params) == input_analysis_field
    assert get_analysis_field_label_from_event_details(params) == input_analysis_field_label
    assert get_analysis_field_unit_from_event_details(params) == input_analysis_field_unit
    assert get_category_field_from_event_details(params) == input_category_field
    assert get_category_field_label_from_event_details(params) == input_category_field_label


def test_get_spatial_features_group(client):
    group_name = "mep"
    group_id = "15698426-7e0f-41df-9bc3-495d87e2e097"
    sfg = get_spatial_features_group(client=client, spatial_features_group_name=group_name)

    assert isinstance(sfg, gpd.GeoDataFrame)
    assert not sfg.empty
    assert "pk" in sfg
    assert "name" in sfg
    assert "short_name" in sfg
    assert "feature_type" in sfg
    assert "geometry" in sfg
    assert "metadata" in sfg
    assert sfg["metadata"].iloc[0]["display_name"] == group_name
    assert sfg["metadata"].iloc[0]["id"] == group_id


def test_get_fields_from_event_type_schema(client):
    fields = client.get_fields_from_event_type_schema(event_type="elephant_sigthing_test")
    assert isinstance(fields, dict)
    assert {
        "name_of_collared_elephant": "Name of collared elephant",
        "herd_type": "Herd Type",
        "how_many_identifiable_individuals": "How many identifiable individuals?",
        "how_many_elephants_are_in_the_group": "How many elephants are in the group?",
    } == fields


def test_get_fields_from_bad_event_type_schema(client):
    from ecoscope.io.earthranger import ERClientNotFound  # type: ignore[import-untyped]

    with pytest.raises(ERClientNotFound):
        client.get_fields_from_event_type_schema(event_type="  ")


# ---------------------------------------------------------------------------
# ERWarehouseClient integration tests
# ---------------------------------------------------------------------------


def _make_observations_arrow_table():
    """Build a minimal pa.Table matching OBSERVATIONS_SCHEMA__ECOSCOPE_SLIM_V1."""
    import geoarrow.pyarrow as ga  # type: ignore[import-untyped]
    import pyarrow as pa
    from shapely.geometry import Point

    wkb_geom = Point(36.8, -1.3).wkb
    return pa.table(
        {
            "geometry": ga.array([wkb_geom]),
            "fixtime": pa.array(
                [pd.Timestamp("2024-01-15", tz="UTC")],
                type=pa.timestamp("ns", tz="UTC"),
            ),
            "groupby_col": pa.array(["subject-1"]),
            "extra__subject__name": pa.array(["Elephant A"]),
            "extra__subject__subject_subtype": pa.array(["elephant"]),
            "junk_status": pa.array([False]),
        },
    )


def _make_patrol_observations_arrow_table():
    """Build a minimal pa.Table matching OBSERVATIONS_WITH_PATROL_SCHEMA_SLIM_V1."""
    import geoarrow.pyarrow as ga  # type: ignore[import-untyped]
    import pyarrow as pa
    from shapely.geometry import Point

    wkb_geom = Point(36.8, -1.3).wkb
    return pa.table(
        {
            "geometry": ga.array([wkb_geom]),
            "fixtime": pa.array(
                [pd.Timestamp("2024-01-15", tz="UTC")],
                type=pa.timestamp("ns", tz="UTC"),
            ),
            "groupby_col": pa.array(["subject-1"]),
            "extra__subject__name": pa.array(["Elephant A"]),
            "extra__subject__subject_subtype": pa.array(["elephant"]),
            "junk_status": pa.array([False]),
            "patrol_id": pa.array(["patrol-1"]),
            "patrol_title": pa.array(["Routine patrol"]),
            "patrol_serial_number": pa.array([1], type=pa.int64()),
            "patrol_status": pa.array(["done"]),
            "patrol_type__value": pa.array(["routine_patrol"]),
            "patrol_type__display": pa.array(["Routine Patrol"]),
            "patrol_start_time": pa.array(["2024-01-15T00:00:00"]),
            "patrol_end_time": pa.array(["2024-01-15T23:59:59"]),
        },
    )


_WAREHOUSE_ENV = {
    "USE_EARTHRANGER_WAREHOUSE_API": "true",
    "EARTHRANGER_WAREHOUSE_API_BASE_URL": "http://warehouse-test",
}


def test_make_warehouse_client_from_env_enabled():
    from ecoscope_earthranger_io_core.client import ERWarehouseClient

    with patch.dict(os.environ, _WAREHOUSE_ENV):
        result = _make_warehouse_client_from_env(er_site_url="mep-dev.pamdas.org", er_api_token="test-token")

    assert isinstance(result, ERWarehouseClient)
    assert result.server == "mep-dev.pamdas.org"
    assert result.warehouse_base_url == "http://warehouse-test"


def test_make_warehouse_client_from_env_missing_url():
    env = {"USE_EARTHRANGER_WAREHOUSE_API": "true"}
    with patch.dict(os.environ, env, clear=True):
        result = _make_warehouse_client_from_env(er_site_url="mep-dev.pamdas.org", er_api_token="test-token")
    assert result is None


def test_get_subjectgroup_observations_via_warehouse_client():
    mock_legacy_client = MagicMock()
    mock_warehouse_client = MagicMock()
    mock_warehouse_client.get_subjectgroup_observations.return_value = _make_observations_arrow_table()

    with patch(
        "ecoscope.platform.tasks.io._earthranger._make_warehouse_client_from_env",
        return_value=mock_warehouse_client,
    ):
        result = get_subjectgroup_observations(
            client=mock_legacy_client,
            subject_group_name="Ecoscope",
            time_range=TimeRange(
                since=datetime.strptime("2024-01-01", "%Y-%m-%d").replace(tzinfo=timezone.utc),
                until=datetime.strptime("2024-03-01", "%Y-%m-%d").replace(tzinfo=timezone.utc),
                timezone=UTC_TIMEZONEINFO,
            ),
            raise_on_empty=False,
        )

    mock_warehouse_client.get_subjectgroup_observations.assert_called_once()
    mock_legacy_client.get_subjectgroup_observations.assert_not_called()
    assert isinstance(result, gpd.GeoDataFrame)
    assert "geometry" in result
    assert "fixtime" in result
    assert "groupby_col" in result
    assert "junk_status" in result


def test_get_patrol_observations_via_warehouse_client():
    mock_legacy_client = MagicMock()
    mock_warehouse_client = MagicMock()
    mock_warehouse_client.get_patrol_observations_with_patrol_filter.return_value = (
        _make_patrol_observations_arrow_table()
    )

    with patch(
        "ecoscope.platform.tasks.io._earthranger._make_warehouse_client_from_env",
        return_value=mock_warehouse_client,
    ):
        result = get_patrol_observations(
            client=mock_legacy_client,
            time_range=TimeRange(
                since=datetime.strptime("2024-01-01", "%Y-%m-%d").replace(tzinfo=timezone.utc),
                until=datetime.strptime("2024-03-01", "%Y-%m-%d").replace(tzinfo=timezone.utc),
                timezone=UTC_TIMEZONEINFO,
            ),
            patrol_types=["routine_patrol"],
            raise_on_empty=False,
        )

    mock_warehouse_client.get_patrol_observations_with_patrol_filter.assert_called_once()
    mock_legacy_client.get_patrol_observations_with_patrol_filter.assert_not_called()
    assert isinstance(result, gpd.GeoDataFrame)
    assert "geometry" in result
    assert "patrol_type__value" in result


def test_get_patrol_observations_from_patrols_df_via_warehouse_client():
    mock_legacy_client = MagicMock()
    mock_warehouse_client = MagicMock()
    mock_warehouse_client.get_patrol_observations.return_value = _make_patrol_observations_arrow_table()
    patrols_df = pd.DataFrame({"id": ["patrol-1"], "state": ["done"]})

    with patch(
        "ecoscope.platform.tasks.io._earthranger._make_warehouse_client_from_env",
        return_value=mock_warehouse_client,
    ):
        result = get_patrol_observations_from_patrols_df(
            client=mock_legacy_client,
            patrols_df=patrols_df,
            raise_on_empty=False,
        )

    mock_warehouse_client.get_patrol_observations.assert_called_once()
    mock_legacy_client.get_patrol_observations.assert_not_called()
    assert isinstance(result, gpd.GeoDataFrame)
    assert "geometry" in result
    assert "patrol_type__value" in result


def test_warehouse_disabled_falls_back_to_legacy_client():
    """When warehouse is not configured, the legacy client should be used."""
    mock_legacy_client = MagicMock()
    mock_legacy_client.get_subjectgroup_observations.return_value = pd.DataFrame()

    with patch(
        "ecoscope.platform.tasks.io._earthranger._make_warehouse_client_from_env",
        return_value=None,
    ):
        result = get_subjectgroup_observations(
            client=mock_legacy_client,
            subject_group_name="Ecoscope",
            time_range=TimeRange(
                since=datetime.strptime("2024-01-01", "%Y-%m-%d").replace(tzinfo=timezone.utc),
                until=datetime.strptime("2024-03-01", "%Y-%m-%d").replace(tzinfo=timezone.utc),
                timezone=UTC_TIMEZONEINFO,
            ),
            raise_on_empty=False,
        )

    mock_legacy_client.get_subjectgroup_observations.assert_called_once()
    assert result.empty
