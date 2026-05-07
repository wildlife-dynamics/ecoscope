import os
from datetime import datetime, timezone

import pytest
from pydantic import TypeAdapter

from ecoscope.platform.connections import SmartConnection
from ecoscope.platform.schemas import (
    EventGDF,
    PatrolObservationsGDF,
)
from ecoscope.platform.tasks.filter._filter import UTC_TIMEZONEINFO, TimeRange
from ecoscope.platform.tasks.io import (
    get_events_from_smart,
    get_patrol_observations_from_smart,
)

pytestmark = pytest.mark.io


@pytest.fixture
def client():
    return SmartConnection(
        server=os.environ["SMART_SERVER"],
        username=os.environ["SMART_USERNAME"],
        password=os.environ["SMART_PASSWORD"],
    ).get_client()


@pytest.mark.skip(reason="SMART API is currently down")
def test_smart_get_patrol_observations(client):
    ta = TypeAdapter(PatrolObservationsGDF)

    observations_relocs = get_patrol_observations_from_smart(
        client=client,
        time_range=TimeRange(
            since=datetime.strptime("2024-01-01", "%Y-%m-%d").replace(tzinfo=timezone.utc),
            until=datetime.strptime("2024-01-02", "%Y-%m-%d").replace(tzinfo=timezone.utc),
            timezone=UTC_TIMEZONEINFO,
        ),
        ca_uuid="735606d2-c34e-49c3-a45b-7496ca834e58",
        language_uuid="13451893-86af-4ec0-beac-2b8e0c2482b5",
    )

    ta.validate_python(observations_relocs)
    assert len(observations_relocs) > 0
    assert "geometry" in observations_relocs
    assert "groupby_col" in observations_relocs
    assert "fixtime" in observations_relocs


@pytest.mark.skip(reason="SMART API is currently down")
def test_smart_get_events(client):
    ta = TypeAdapter(EventGDF)

    result = get_events_from_smart(
        client=client,
        time_range=TimeRange(
            since=datetime.strptime("2024-01-01", "%Y-%m-%d").replace(tzinfo=timezone.utc),
            until=datetime.strptime("2024-01-02", "%Y-%m-%d").replace(tzinfo=timezone.utc),
            timezone=UTC_TIMEZONEINFO,
        ),
        ca_uuid="735606d2-c34e-49c3-a45b-7496ca834e58",
        language_uuid="13451893-86af-4ec0-beac-2b8e0c2482b5",
    )

    ta.validate_python(result)
    assert len(result) > 0
    assert "uuid" in result
    assert "time" in result
    assert "event_type" in result
    assert "geometry" in result
