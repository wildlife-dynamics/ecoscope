import pandas as pd

from ecoscope.platform.indexes import AllGrouper, ValueGrouper
from ecoscope.platform.tasks.transformation import explode, explode_for_groupers


def test_explode():
    df = pd.DataFrame(
        {
            "id": ["id1", "id2", "id3", "id4"],
            "column": [["a", "b", "c"], ["a"], [], ["c", "d", "d", "f"]],
        }
    )

    exploded_df = explode(df, "column", ignore_index=True)
    expected_df = pd.DataFrame(
        {
            "id": ["id1", "id1", "id1", "id2", "id3", "id4", "id4", "id4", "id4"],
            "column": ["a", "b", "c", "a", None, "c", "d", "d", "f"],
        }
    )
    pd.testing.assert_frame_equal(exploded_df, expected_df)


def test_explode_for_groupers_explodes_only_selected_list_columns():
    df = pd.DataFrame(
        {
            "patrol_member_names": [["Asha", "Ben"], ["Chidi"]],
            "patrol_asset_names": [["LC"], ["LC"]],
            "dist_meters": [12000.0, 5000.0],
        }
    )
    out = explode_for_groupers(df, [ValueGrouper(index_name="patrol_member_names")])
    assert out["patrol_member_names"].tolist() == ["Asha", "Ben", "Chidi"]
    assert out["dist_meters"].tolist() == [12000.0, 12000.0, 5000.0]
    assert out["patrol_asset_names"].tolist() == [["LC"], ["LC"], ["LC"]]


def test_explode_for_groupers_empty_list_keeps_one_row():
    df = pd.DataFrame({"patrol_member_names": [[]], "dist_meters": [1.0]})
    out = explode_for_groupers(df, [ValueGrouper(index_name="patrol_member_names")])
    assert len(out) == 1 and out["patrol_member_names"].tolist() == ["None"]


def test_explode_for_groupers_is_noop_for_all_grouper_scalar_and_absent_columns():
    df = pd.DataFrame({"patrol_type": ["a", "b"], "patrol_member_names": [["x"], ["y", "z"]]})
    assert explode_for_groupers(df, AllGrouper()).equals(df)
    assert explode_for_groupers(df, [ValueGrouper(index_name="patrol_type")]).equals(df)
    assert explode_for_groupers(df, [ValueGrouper(index_name="missing")]).equals(df)


def test_explode_for_groupers_preserves_multiindex_levels():
    df = pd.DataFrame({"patrol_member_names": [["a", "b"]], "v": [1]}).set_index(
        pd.Index(["2026-01"], name="TemporalGrouper_%Y-%m"), append=True
    )
    out = explode_for_groupers(df, [ValueGrouper(index_name="patrol_member_names")])
    assert out.index.names == df.index.names and len(out) == 2
