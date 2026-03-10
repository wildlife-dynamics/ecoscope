import pandas as pd

from ecoscope.platform.tasks.transformation import explode


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
