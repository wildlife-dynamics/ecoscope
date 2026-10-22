import pandas as pd

from ecoscope.platform.tasks.transformation._leg_names import add_leg_names_to_tracks

legs = pd.DataFrame(
    {
        "patrol_segment_id": ["s1", "s2"],
        "patrol_team_name": ["Alpha", "None"],
        "patrol_member_names": [["Ann", "Bob"], ["Cy"]],
        "patrol_asset_names": [["Truck"], []],
        "leg_active_seconds": [7200.0, 3600.0],
    }
)
tracks = pd.DataFrame(
    {"patrol_segment_id": ["s1", "s1", "s2", "s9"], "dist_meters": [1.0, 2.0, 3.0, 4.0]},
    index=[10, 11, 12, 13],
)


def test_names_copied_by_segment_id_without_duplicating_rows():
    out = add_leg_names_to_tracks(tracks, legs)
    assert len(out) == 4 and list(out.index) == [10, 11, 12, 13]
    assert out["patrol_team_name"].tolist() == ["Alpha", "Alpha", "None", "None"]
    assert out["patrol_member_names"].tolist()[:3] == [["Ann", "Bob"], ["Ann", "Bob"], ["Cy"]]
    assert out["patrol_asset_names"].tolist()[2] == []
    assert out.loc[13, "patrol_member_names"] == "None" and out.loc[13, "patrol_asset_names"] == "None"
    assert out["dist_meters"].tolist() == [1.0, 2.0, 3.0, 4.0]


def test_frame_without_segment_id_gets_none_everywhere():
    out = add_leg_names_to_tracks(pd.DataFrame({"dist_meters": [1.0, 2.0]}), legs)
    assert out["patrol_team_name"].tolist() == ["None", "None"]
    assert out["patrol_member_names"].tolist() == ["None", "None"]
    assert out["patrol_asset_names"].tolist() == ["None", "None"]


def test_empty_legs_gives_none_and_missing_name_column_gives_none():
    out = add_leg_names_to_tracks(tracks, legs.iloc[0:0])
    assert out["patrol_team_name"].tolist() == ["None"] * 4
    partial = add_leg_names_to_tracks(tracks, legs.drop(columns=["patrol_asset_names"]))
    assert partial["patrol_asset_names"].tolist() == ["None"] * 4
    assert partial["patrol_team_name"].tolist()[0] == "Alpha"


def test_input_frame_is_not_mutated():
    before = tracks.copy()
    add_leg_names_to_tracks(tracks, legs)
    assert tracks.equals(before)
