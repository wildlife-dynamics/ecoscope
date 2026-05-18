import os
from datetime import datetime, timezone
from unittest.mock import MagicMock, call, patch

import geopandas as gpd  # type: ignore[import-untyped]
import pandas as pd
import pytest
from pydantic import SecretStr
from wt_task import task

from ecoscope.platform.connections import EarthRangerClientProtocol, EarthRangerConnection
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


def test_get_patrol_observations():
    """The task-validated entry point strips whitespace from patrol_types before the
    underlying client call. This is the wrapper behavior we want to lock in."""
    mock_client = _make_mock_client()
    mock_client.get_patrol_observations_with_patrol_filter.return_value = pd.DataFrame()

    with (
        _patched_named_connection(mock_client),
        patch(
            "ecoscope.platform.tasks.io._earthranger._make_warehouse_client_from_env",
            return_value=None,
        ),
    ):
        task(get_patrol_observations).validate().call(
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
            raise_on_empty=False,
        )

    call_kwargs = mock_client.get_patrol_observations_with_patrol_filter.call_args.kwargs
    assert call_kwargs["patrol_type_value"] == ["ecoscope_patrol"]


def test_get_patrol_observations_with_whitespace_in_patrol_types_without_pydantic_validations():
    """Direct (non-task) calls bypass pydantic, so whitespace is passed through as-is to
    the underlying client. This is the complement to the test above: stripping is
    pydantic's responsibility, not the wrapper's."""
    mock_client = _make_mock_client()
    mock_client.get_patrol_observations_with_patrol_filter.return_value = pd.DataFrame()

    with patch(
        "ecoscope.platform.tasks.io._earthranger._make_warehouse_client_from_env",
        return_value=None,
    ):
        get_patrol_observations(
            client=mock_client,
            time_range=TimeRange(
                since=datetime.strptime("2015-01-01", "%Y-%m-%d").replace(tzinfo=timezone.utc),
                until=datetime.strptime("2015-03-01", "%Y-%m-%d").replace(tzinfo=timezone.utc),
                timezone=UTC_TIMEZONEINFO,
            ),
            patrol_types=["    ecoscope_patrol           "],
            status=None,
            include_patrol_details=True,
            raise_on_empty=False,
        )

    call_kwargs = mock_client.get_patrol_observations_with_patrol_filter.call_args.kwargs
    assert call_kwargs["patrol_type_value"] == ["    ecoscope_patrol           "]


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


def test_get_events_with_event_type_whitespace():
    """Task-validated event_types are stripped of whitespace before the wrapper's
    get_event_types lookup. Verify by mocking get_event_types to return clean values
    and confirming the wrapper resolved IDs (i.e., stripping happened, otherwise
    the .isin() lookup would miss everything)."""
    expected_clean = [
        "hwc_rep",
        "bird_sighting_rep",
        "wildlife_sighting_rep",
        "poacher_camp_rep",
        "fire_rep",
        "injured_animal_rep",
    ]
    event_types_df = pd.DataFrame(
        {
            "id": [f"id-{i}" for i in range(len(expected_clean))],
            "value": expected_clean,
        }
    )
    mock_client = _make_mock_client()
    mock_client.get_event_types.return_value = event_types_df
    mock_client.get_events.return_value = pd.DataFrame(
        {"id": ["e1"], "time": [pd.Timestamp("2015-06-01", tz="UTC")], "event_type": ["hwc_rep"], "geometry": [None]}
    )

    with _patched_named_connection(mock_client):
        task(get_events).validate().call(
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
                "   bird_sighting_rep           ",
                "      wildlife_sighting_rep        ",
                "  poacher_camp_rep    ",
                "    fire_rep   ",
                "     injured_animal_rep   ",
            ],
            event_columns=["id", "time", "event_type", "geometry"],
        )

    # If stripping happened, all 6 ids should be resolved and passed to client.get_events.
    assert mock_client.get_events.call_count == 1
    resolved_ids = mock_client.get_events.call_args.kwargs["event_type"]
    assert sorted(resolved_ids) == sorted(event_types_df["id"].tolist())


def test_get_events_bad_event_type():
    """When event_types don't match any known type, the wrapper short-circuits to an
    empty DataFrame without calling get_events."""
    mock_client = _make_mock_client()
    # The known event types in this fake registry don't include "not a real type"
    mock_client.get_event_types.return_value = pd.DataFrame({"id": ["abc123"], "value": ["a_real_type"]})

    result = get_events(
        client=mock_client,
        time_range=TimeRange(
            since=datetime.strptime("2015-01-01", "%Y-%m-%d").replace(tzinfo=timezone.utc),
            until=datetime.strptime("2015-12-31", "%Y-%m-%d").replace(tzinfo=timezone.utc),
            timezone=UTC_TIMEZONEINFO,
        ),
        event_types=["not a real type"],
        raise_on_empty=False,
        event_columns=["id", "time", "event_type", "geometry"],
    )

    assert result.empty
    mock_client.get_events.assert_not_called()


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


@pytest.mark.parametrize("force_point_geometry", [True, False])
def test_get_events_forwards_force_point_geometry(mock_empty_client, force_point_geometry):
    get_events(
        client=mock_empty_client,
        time_range=TimeRange(
            since=datetime.strptime("2017-01-01", "%Y-%m-%d").replace(tzinfo=timezone.utc),
            until=datetime.strptime("2017-03-31", "%Y-%m-%d").replace(tzinfo=timezone.utc),
            timezone=UTC_TIMEZONEINFO,
        ),
        event_types=["an_event"],
        raise_on_empty=False,
        force_point_geometry=force_point_geometry,
    )
    assert mock_empty_client.get_events.call_args.kwargs["force_point_geometry"] is force_point_geometry


def test_get_events_force_point_geometry_default_is_true(mock_empty_client):
    get_events(
        client=mock_empty_client,
        time_range=TimeRange(
            since=datetime.strptime("2017-01-01", "%Y-%m-%d").replace(tzinfo=timezone.utc),
            until=datetime.strptime("2017-03-31", "%Y-%m-%d").replace(tzinfo=timezone.utc),
            timezone=UTC_TIMEZONEINFO,
        ),
        event_types=["an_event"],
        raise_on_empty=False,
    )
    assert mock_empty_client.get_events.call_args.kwargs["force_point_geometry"] is True


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


def test_patrol_events_combined():
    """The set_patrols_and_patrol_events_params -> _from_combined_params path produces
    the same underlying IO calls as the direct task path, for both patrol-observations
    and patrol-events."""
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

    mock_client = _make_mock_client()
    mock_client.get_patrol_observations_with_patrol_filter.return_value = pd.DataFrame()
    mock_client.get_patrol_events.return_value = pd.DataFrame()

    with (
        _patched_named_connection(mock_client),
        patch(
            "ecoscope.platform.tasks.io._earthranger._make_warehouse_client_from_env",
            return_value=None,
        ),
    ):
        task(get_patrol_observations).validate().call(**patrol_obs_args)
        task(get_patrol_events).validate().call(**patrol_events_args)

        combined_params = set_patrols_and_patrol_events_params(**combined_args)
        get_patrol_events_from_combined_params(combined_params)
        get_patrol_observations_from_combined_params(combined_params)

    # Each underlying method should have been called twice (once per code path) with
    # identical args. assert_has_calls is order-preserving but allows other calls
    # in between, which is fine here.
    mock_client.get_patrol_observations_with_patrol_filter.assert_has_calls(
        [
            call(**expected_patrol_obs_call_args),
            call(**expected_patrol_obs_call_args),
        ]
    )
    mock_client.get_patrol_events.assert_has_calls(
        [
            call(**expected_patrol_events_call_args),
            call(**expected_patrol_events_call_args),
        ]
    )


# The *_combined_parity tests verify that the "task-style" entry point (e.g.
# `task(get_patrol_observations).validate().call(...)`) and the "combined-params"
# entry point (e.g. `get_patrol_observations_from_combined_params(...)`) produce
# identical underlying IO calls. That's a logical equivalence check, so it doesn't
# need a live EarthRanger server. The connection layer and warehouse selector are
# mocked so the test is fully hermetic — no login, no downloads, no flakes.


def _assert_calls_match(calls):
    """Assert that a mock was invoked exactly twice with identical args.

    Handles DataFrames in kwargs (which can't be compared with ==).
    """
    assert len(calls) == 2, f"Expected 2 calls, got {len(calls)}"
    a, b = calls[0], calls[1]
    assert a.args == b.args, f"positional args differ: {a.args} vs {b.args}"
    assert a.kwargs.keys() == b.kwargs.keys(), f"kwarg keys differ: {a.kwargs.keys()} vs {b.kwargs.keys()}"
    for k in a.kwargs:
        va, vb = a.kwargs[k], b.kwargs[k]
        if isinstance(va, pd.DataFrame):
            pd.testing.assert_frame_equal(va, vb)
        else:
            assert va == vb, f"kwarg {k!r} differs: {va!r} vs {vb!r}"


def _make_mock_client():
    """Build a MagicMock that satisfies the pydantic protocol check AND can stand in
    for an EarthRangerIO instance when callers access `.server` / `.token` (the
    warehouse selector reads these before any IO method is called)."""
    mock_client = MagicMock(spec=EarthRangerClientProtocol)
    mock_client.server = "fake-server"
    mock_client.token = None
    return mock_client


def _patched_named_connection(mock_client):
    """Make `EarthRangerConnection.client_from_named_connection(<any>)` return mock_client.

    Patching at `from_named_connection` (rather than at `client_from_named_connection`)
    works around the fact that the BeforeValidator captures the bound classmethod at
    import time — but `from_named_connection` is looked up dynamically inside that
    classmethod, so patching it at runtime takes effect. As a side effect, no env
    vars are required: env reading happens inside the patched-out method.
    """
    fake_conn = MagicMock()
    fake_conn.get_client.return_value = mock_client
    return patch(
        "ecoscope.platform.connections.EarthRangerConnection.from_named_connection",
        return_value=fake_conn,
    )


def _fake_patrols_df():
    """Minimal DataFrame satisfying PatrolsDFSchema (id, state, serial_number, patrol_segments)."""
    return pd.DataFrame(
        {
            "id": ["patrol-1"],
            "state": ["done"],
            "serial_number": pd.array([1], dtype="int64"),
            "patrol_segments": [[]],
        }
    )


def test_get_patrol_observations_combined_parity():
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
        "raise_on_empty": False,
    }
    mock_client = _make_mock_client()
    mock_client.get_patrol_observations_with_patrol_filter.return_value = pd.DataFrame()

    with (
        _patched_named_connection(mock_client),
        patch(
            "ecoscope.platform.tasks.io._earthranger._make_warehouse_client_from_env",
            return_value=None,
        ),
    ):
        task(get_patrol_observations).validate().call(**args)
        # event_types is not an arg in get_patrol_observations, but it is required for
        # CombinedPatrolAndEventsParams so explicitly add here
        combined = CombinedPatrolAndEventsParams(**args, event_types=[])
        get_patrol_observations_from_combined_params(combined)

    _assert_calls_match(mock_client.get_patrol_observations_with_patrol_filter.call_args_list)


def test_get_patrol_events_combined_parity():
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
        "raise_on_empty": False,
    }
    mock_client = _make_mock_client()
    mock_client.get_patrol_events.return_value = pd.DataFrame()

    with _patched_named_connection(mock_client):
        task(get_patrol_events).validate().call(**args)
        combined = CombinedPatrolAndEventsParams(**args)
        get_patrol_events_from_combined_params(combined)

    _assert_calls_match(mock_client.get_patrol_events.call_args_list)


def test_get_patrols_combined_parity():
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
        "raise_on_empty": False,
    }
    mock_client = _make_mock_client()
    mock_client.get_patrols.return_value = pd.DataFrame()

    with _patched_named_connection(mock_client):
        task(get_patrols).validate().call(**args)
        # event_types is not an arg in get_patrols, but it is required for
        # CombinedPatrolAndEventsParams so explicitly add here
        combined = CombinedPatrolAndEventsParams(**args, event_types=[])
        get_patrols_from_combined_params(combined)

    _assert_calls_match(mock_client.get_patrols.call_args_list)


def test_get_patrol_observations_from_patrols_df_combined_parity():
    time_range = TimeRange(
        since=datetime.strptime("2015-01-01", "%Y-%m-%d").replace(tzinfo=timezone.utc),
        until=datetime.strptime("2015-03-01", "%Y-%m-%d").replace(tzinfo=timezone.utc),
        timezone=UTC_TIMEZONEINFO,
    )
    patrol_types = "ecoscope_patrol"
    status = None
    patrols_df = _fake_patrols_df()

    shared_args = {
        "client": "MEP_DEV",
        "include_patrol_details": True,
        "raise_on_empty": False,
    }
    mock_client = _make_mock_client()
    mock_client.get_patrol_observations.return_value = pd.DataFrame()

    with (
        _patched_named_connection(mock_client),
        patch(
            "ecoscope.platform.tasks.io._earthranger._make_warehouse_client_from_env",
            return_value=None,
        ),
    ):
        task(get_patrol_observations_from_patrols_df).validate().call(**shared_args, patrols_df=patrols_df)
        combined = CombinedPatrolAndEventsParams(
            **shared_args,
            time_range=time_range,
            patrol_types=patrol_types,
            status=status,
            event_types=[],
        )
        get_patrol_observations_from_patrols_df_and_combined_params(patrols_df=patrols_df, combined_params=combined)

    _assert_calls_match(mock_client.get_patrol_observations.call_args_list)


def test_unpack_events_from_patrols_df_combined_parity():
    time_range = TimeRange(
        since=datetime.strptime("2015-01-01", "%Y-%m-%d").replace(tzinfo=timezone.utc),
        until=datetime.strptime("2015-03-01", "%Y-%m-%d").replace(tzinfo=timezone.utc),
        timezone=UTC_TIMEZONEINFO,
    )
    patrols_df = _fake_patrols_df()
    shared_args = {
        "time_range": time_range,
        "event_types": [],
        "raise_on_empty": False,
    }
    mock_helper = MagicMock(return_value=pd.DataFrame())

    with patch("ecoscope.io.earthranger_utils.unpack_events_from_patrols_df", mock_helper):
        task(unpack_events_from_patrols_df).validate().call(**shared_args, patrols_df=patrols_df)
        combined = CombinedPatrolAndEventsParams(
            **shared_args,
            client="MEP_DEV",
            patrol_types="ecoscope_patrol",
            status=None,
        )
        unpack_events_from_patrols_df_and_combined_params(patrols_df=patrols_df, combined_params=combined)

    _assert_calls_match(mock_helper.call_args_list)


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


def test_get_patrol_events_with_display_values_empty():
    """When the underlying client returns empty, include_display_values=True must not
    crash and must not call the display-name resolver."""
    mock_client = _make_mock_client()
    mock_client.get_patrol_events.return_value = pd.DataFrame()

    result = get_patrol_events(
        client=mock_client,
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
    mock_client.get_event_type_display_names_from_events.assert_not_called()


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


def test_get_events_with_display_values_empty():
    """When get_events returns empty, include_display_values=True must not crash and
    must not call the display-name resolver."""
    mock_client = _make_mock_client()
    mock_client.get_events.return_value = pd.DataFrame()

    result = get_events(
        client=mock_client,
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
    mock_client.get_event_type_display_names_from_events.assert_not_called()


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
        result = _make_warehouse_client_from_env(er_site_url="mep-dev.pamdas.org", er_api_token=SecretStr("test-token"))

    assert isinstance(result, ERWarehouseClient)
    assert result.server == "mep-dev.pamdas.org"
    assert result.warehouse_base_url == "http://warehouse-test"


def test_make_warehouse_client_from_env_missing_url():
    from ecoscope_earthranger_io_core.client import ERWarehouseClient

    env = {"USE_EARTHRANGER_WAREHOUSE_API": "true"}
    with patch.dict(os.environ, env, clear=True):
        result = _make_warehouse_client_from_env(er_site_url="mep-dev.pamdas.org", er_api_token=SecretStr("test-token"))
    assert isinstance(result, ERWarehouseClient)
    assert result.warehouse_base_url is None


def test_make_warehouse_client_from_env_no_token():
    env = {"USE_EARTHRANGER_WAREHOUSE_API": "true"}
    with patch.dict(os.environ, env, clear=True):
        result = _make_warehouse_client_from_env(er_site_url="mep-dev.pamdas.org", er_api_token=None)
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
