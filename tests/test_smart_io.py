import pytest

pytestmark = pytest.mark.smart_io


def test_smart_get_events(smart_io):
    events = smart_io.get_events(
        ca_uuid="735606d2-c34e-49c3-a45b-7496ca834e58",
        language_uuid="13451893-86af-4ec0-beac-2b8e0c2482b5",
        start="2021-12-20",
        end="2021-12-22",
    )
    assert not events.empty
    assert "time" in events.columns
    assert "event_type" in events.columns
    assert "geometry" in events.columns
    assert "extracted_attributes" in events.columns


def test_smart_get_patrol_observations(smart_io):
    result = smart_io.get_patrol_observations(
        ca_uuid="735606d2-c34e-49c3-a45b-7496ca834e58",
        language_uuid="13451893-86af-4ec0-beac-2b8e0c2482b5",
        start="2021-12-20",
        end="2021-12-22",
    )

    assert len(result.gdf) > 0
    assert "geometry" in result.gdf
    assert "groupby_col" in result.gdf
    assert "fixtime" in result.gdf


def test_smart_get_patrol_observations_paginates(smart_io):
    result = smart_io.get_patrol_observations(
        ca_uuid="735606d2-c34e-49c3-a45b-7496ca834e58",
        language_uuid="13451893-86af-4ec0-beac-2b8e0c2482b5",
        start="2021-12-01",
        end="2021-12-07",
        window_size_in_days=2,
    )

    assert len(result.gdf) > 0
    assert "geometry" in result.gdf
    assert "groupby_col" in result.gdf
    assert "fixtime" in result.gdf
