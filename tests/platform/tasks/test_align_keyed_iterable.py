import pandas as pd
import pytest

from ecoscope.platform.tasks.transformation import align_keyed_iterable_to_reference

JAN = (("month", "=", "jan"),)
FEB = (("month", "=", "feb"),)
MAR = (("month", "=", "mar"),)


@pytest.fixture
def events_df():
    return pd.DataFrame({"event_type": ["fire", "snare"], "count": [1, 2]})


def test_fills_missing_keys_with_empty_schema_match(events_df):
    target = [(JAN, events_df), (FEB, events_df.copy())]
    reference = [(JAN, "traj"), (FEB, "traj"), (MAR, "traj")]

    result = align_keyed_iterable_to_reference(target=target, reference=reference)

    assert [key for key, _ in result] == [JAN, FEB, MAR]
    assert result[0][1] is events_df  # present keys pass through untouched
    mar_df = result[2][1]
    assert len(mar_df) == 0
    assert mar_df.columns.tolist() == events_df.columns.tolist()


def test_result_follows_reference_order(events_df):
    target = [(FEB, events_df)]
    reference = [(MAR, None), (JAN, None), (FEB, None)]

    result = align_keyed_iterable_to_reference(target=target, reference=reference)

    assert [key for key, _ in result] == [MAR, JAN, FEB]


def test_target_only_keys_are_kept(events_df):
    target = [(JAN, events_df), (MAR, events_df.copy())]
    reference = [(JAN, None)]

    result = align_keyed_iterable_to_reference(target=target, reference=reference)

    assert [key for key, _ in result] == [JAN, MAR]
    assert len(result[1][1]) == 2


def test_explicit_fill_value(events_df):
    fill = pd.DataFrame({"custom": []})
    target = [(JAN, events_df)]
    reference = [(JAN, None), (FEB, None)]

    result = align_keyed_iterable_to_reference(target=target, reference=reference, fill_value=fill)

    assert result[1][1].columns.tolist() == ["custom"]


def test_empty_target_without_fill_value_raises(events_df):
    with pytest.raises(ValueError, match="fill_value"):
        align_keyed_iterable_to_reference(target=[], reference=[(JAN, None), (FEB, None)])


def test_empty_target_with_fill_value_fills_all_keys(events_df):
    result = align_keyed_iterable_to_reference(
        target=[], reference=[(JAN, None), (FEB, None)], fill_value=events_df.iloc[0:0]
    )

    assert [key for key, _ in result] == [JAN, FEB]
    assert all(len(df) == 0 for _, df in result)
    assert all(df.columns.tolist() == events_df.columns.tolist() for _, df in result)


def test_duplicate_target_keys_raise(events_df):
    with pytest.raises(ValueError, match="Duplicate key"):
        align_keyed_iterable_to_reference(target=[(JAN, events_df), (JAN, events_df.copy())], reference=[(JAN, None)])


def test_duplicate_reference_keys_deduped(events_df):
    result = align_keyed_iterable_to_reference(
        target=[(JAN, events_df)], reference=[(JAN, None), (JAN, None), (FEB, None)]
    )

    assert [key for key, _ in result] == [JAN, FEB]


def test_fills_are_independent_copies(events_df):
    result = align_keyed_iterable_to_reference(
        target=[(JAN, events_df)],
        reference=[(JAN, None), (FEB, None), (MAR, None)],
    )

    feb_df, mar_df = result[1][1], result[2][1]
    feb_df["new_col"] = []
    assert "new_col" not in mar_df.columns
