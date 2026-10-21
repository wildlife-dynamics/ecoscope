"""Mock-only tests for patrol leg ingest on `EarthRangerIO`. No network login: the
client is built with `__new__` and every method that would reach the server is patched.
"""

import json
from unittest.mock import patch

import geopandas as gpd
import pandas as pd
import pytest
from erclient import ERClientNotFound
from shapely.geometry import Point

import ecoscope
from ecoscope.io.earthranger import EarthRangerIO


@pytest.fixture
def er_io():
    return EarthRangerIO.__new__(EarthRangerIO)


def _leg(seg_id, **extra):
    return {
        "id": seg_id,
        "leader": {"id": "s1", "name": "Asha"},
        "time_range": {"start_time": "2024-01-01T00:00:00+00:00", "end_time": "2024-01-02T00:00:00+00:00"},
        "patrol_type": "foot",
        **extra,
    }


def _leg_patrols_df(segments):
    return pd.DataFrame(
        [{"id": "p1", "title": "Patrol", "serial_number": 1, "state": "done", "patrol_segments": segments}]
    )


def _leg_patrol_types():
    return pd.DataFrame(
        [{"id": "pt1", "value": "foot", "ordernum": 1, "icon_id": "i", "default_priority": 0, "is_active": True}]
    ).set_index("id")


def _leg_observations(*args, **kwargs):
    gdf = gpd.GeoDataFrame(
        {
            "fixtime": pd.to_datetime(["2024-01-01T01:00:00", "2024-01-01T02:00:00"], utc=True),
            "junk_status": [False, False],
            "geometry": [Point(0, 0), Point(1, 1)],
        },
        index=pd.Index(["o1", "o2"], name="id"),
        crs=4326,
    )
    return ecoscope.Relocations.from_gdf(gdf)


LEG_NEW_FIELDS = {
    "team": "t1",
    "members": ["s1", "s2"],
    "assets": ["v1"],
    "segment_details": {"weather": "dry"},
    "type_details": {},
}


@patch("ecoscope.io.earthranger.EarthRangerIO.get_patrol_types")
@patch("ecoscope.io.earthranger.EarthRangerIO.get_subject_observations")
def test_get_patrol_observations_carries_leg_fields(obs_mock, pt_mock, er_io):
    obs_mock.side_effect = _leg_observations
    pt_mock.return_value = _leg_patrol_types()
    patrols_df = _leg_patrols_df(
        [_leg("seg-a", **LEG_NEW_FIELDS, is_pause=False), _leg("seg-b", **LEG_NEW_FIELDS, is_pause=True)]
    )

    gdf = er_io.get_patrol_observations(patrols_df, include_patrol_details=True).gdf

    assert set(gdf["patrol_segment_id"]) == {"seg-a"}
    assert gdf["patrol_team"].iloc[0] == "t1"
    assert gdf["patrol_members"].iloc[0] == ["s1", "s2"]
    assert gdf["patrol_assets"].iloc[0] == ["v1"]
    assert json.loads(gdf["patrol_segment_details"].iloc[0]) == {"weather": "dry"}
    assert json.loads(gdf["patrol_type_details"].iloc[0]) == {}


@patch("ecoscope.io.earthranger.EarthRangerIO.get_patrol_types")
@patch("ecoscope.io.earthranger.EarthRangerIO.get_subject_observations")
def test_get_patrol_observations_skips_pause_legs(obs_mock, pt_mock, er_io):
    obs_mock.side_effect = _leg_observations
    pt_mock.return_value = _leg_patrol_types()
    patrols_df = _leg_patrols_df(
        [_leg("seg-a", **LEG_NEW_FIELDS, is_pause=False), _leg("seg-b", **LEG_NEW_FIELDS, is_pause=True)]
    )

    er_io.get_patrol_observations(patrols_df, include_patrol_details=True)

    assert obs_mock.call_count == 1


@patch("ecoscope.io.earthranger.EarthRangerIO.get_patrol_types")
@patch("ecoscope.io.earthranger.EarthRangerIO.get_subject_observations")
def test_get_patrol_observations_pause_leg_with_unknown_type_is_skipped(obs_mock, pt_mock, er_io):
    obs_mock.side_effect = _leg_observations
    pt_mock.return_value = _leg_patrol_types()
    patrols_df = _leg_patrols_df([_leg("seg-a"), _leg("seg-p", is_pause=True, patrol_type="unknown")])

    gdf = er_io.get_patrol_observations(patrols_df, include_patrol_details=True).gdf

    assert set(gdf["patrol_segment_id"]) == {"seg-a"}
    assert obs_mock.call_count == 1


@patch("ecoscope.io.earthranger.EarthRangerIO.get_patrol_types")
@patch("ecoscope.io.earthranger.EarthRangerIO.get_subject_observations")
def test_get_patrol_observations_details_frame_writes_to_parquet(obs_mock, pt_mock, er_io, tmp_path):
    obs_mock.side_effect = _leg_observations
    pt_mock.return_value = _leg_patrol_types()
    patrols_df = _leg_patrols_df([_leg("seg-a", **LEG_NEW_FIELDS)])

    gdf = er_io.get_patrol_observations(patrols_df, include_patrol_details=True).gdf

    gdf[["patrol_segment_details", "patrol_type_details"]].to_parquet(tmp_path / "details.parquet")


@patch("ecoscope.io.earthranger.EarthRangerIO.get_patrol_types")
@patch("ecoscope.io.earthranger.EarthRangerIO.get_subject_observations")
def test_get_patrol_observations_without_new_fields_adds_no_columns(obs_mock, pt_mock, er_io):
    obs_mock.side_effect = _leg_observations
    pt_mock.return_value = _leg_patrol_types()
    patrols_df = _leg_patrols_df([_leg("seg-a"), _leg("seg-b")])

    gdf = er_io.get_patrol_observations(patrols_df, include_patrol_details=True).gdf

    assert "patrol_segment_id" in gdf.columns
    for col in ["patrol_team", "patrol_members", "patrol_assets", "patrol_segment_details", "patrol_type_details"]:
        assert col not in gdf.columns


@patch("ecoscope.io.earthranger.EarthRangerIO._get")
def test_get_patrol_teams(get_mock, er_io):
    get_mock.return_value = {
        "teams": [
            {"id": "t1", "value": "alpha", "display": "Alpha", "is_active": True},
            {"id": "t2", "value": "bravo", "display": "Bravo", "is_active": False},
        ]
    }
    teams = er_io.get_patrol_teams()
    assert teams[["id", "name"]].values.tolist() == [["t1", "Alpha"], ["t2", "Bravo"]]
    assert get_mock.call_args.args[0] == "activity/patrols/config/default/"


@patch("ecoscope.io.earthranger.EarthRangerIO._get")
def test_get_patrol_teams_empty_display_uses_value(get_mock, er_io):
    get_mock.return_value = {"teams": [{"id": "t1", "value": "alpha", "display": ""}, {"id": "t2", "value": "bravo"}]}
    teams = er_io.get_patrol_teams()
    assert teams.values.tolist() == [["t1", "alpha"], ["t2", "bravo"]]


@patch("ecoscope.io.earthranger.EarthRangerIO._get", side_effect=ERClientNotFound())
def test_get_patrol_teams_404_is_empty(get_mock, er_io):
    teams = er_io.get_patrol_teams()
    assert teams.empty and list(teams.columns) == ["id", "name"]
