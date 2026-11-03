"""Tasks taking `EarthRangerSource` must still accept a bare API client.

Ext task libraries call these tasks directly with an `EarthRangerClient`, which skips
the validator that would resolve it to `EarthRangerBackends`.
"""

import inspect
from typing import get_type_hints
from unittest.mock import MagicMock

import geopandas as gpd  # type: ignore[import-untyped]
import pytest
from shapely.geometry import Point

from ecoscope.io import EarthRangerIO
from ecoscope.platform.connections import EarthRangerSource
from ecoscope.platform.tasks.io import _earthranger, get_fields_from_event_type_schema, get_spatial_features_group

_SOURCE_TASKS = [
    f
    for name, f in inspect.getmembers(_earthranger, inspect.isfunction)
    if not name.startswith("_")
    and f.__module__ == _earthranger.__name__
    and get_type_hints(f, include_extras=True).get("client") == EarthRangerSource
]


def _bare_client():
    api = MagicMock(spec=EarthRangerIO)
    api.server = "https://mep-dev.pamdas.org"
    api.token = None
    return api


def test_source_tasks_are_found():
    assert len(_SOURCE_TASKS) >= 12


@pytest.mark.parametrize("f", _SOURCE_TASKS, ids=lambda f: f.__name__)
def test_source_task_normalizes_its_client(f):
    """Guards new tasks too: anything reading `client.api`/`client.warehouse` must normalize first."""
    assert "client = as_backends(client)" in inspect.getsource(f)


def test_get_spatial_features_group_with_bare_client():
    api = _bare_client()
    api.get_spatial_features_group.return_value = {
        "id": "sfg-1",
        "name": "g",
        "description": "",
        "url": "",
        "feature_count": 1,
        "created_at": "",
        "updated_at": "",
        "features": gpd.GeoDataFrame(geometry=[Point(0, 0)], crs="EPSG:4326"),
    }

    with pytest.deprecated_call():
        regions = get_spatial_features_group(client=api, spatial_features_group_name="g")

    assert regions["metadata"].tolist() == [{"id": "sfg-1", "display_name": "g"}]


def test_get_fields_from_event_type_schema_with_bare_client():
    api = _bare_client()
    api.get_fields_from_event_type_schema.return_value = {"a": "A"}

    with pytest.deprecated_call():
        assert get_fields_from_event_type_schema(client=api, event_type="x") == {"a": "A"}
