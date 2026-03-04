import pytest
from ecoscope.platform.indexes import (
    AllGrouper,
    Date,
    DayOfTheMonth,
    DayOfTheWeek,
    DayOfTheYear,
    Hour,
    Month,
    SpatialGrouper,
    TemporalGrouper,
    ValueGrouper,
    Year,
    YearMonth,
)


def test_sort_key_is_none():
    """These temporal index classes should have a sort_key of None, because they are
    string digits (i.e. numeric), and Python's default sort() method will work fine.
    """
    assert Year.sort_key is None
    assert YearMonth.sort_key is None
    assert DayOfTheYear.sort_key is None
    assert DayOfTheMonth.sort_key is None
    assert Date.sort_key is None


@pytest.mark.parametrize(
    "month, expected_index",
    [
        ("January", 1),
        ("February", 2),
        ("March", 3),
        ("April", 4),
        ("May", 5),
        ("June", 6),
        ("July", 7),
        ("August", 8),
        ("September", 9),
        ("October", 10),
        ("November", 11),
        ("December", 12),
    ],
)
def test_month_sort_key(month: str, expected_index: int):
    assert Month.sort_key(month) == expected_index


@pytest.mark.parametrize(
    "day_of_week, expected_index",
    [
        ("Monday", 0),
        ("Tuesday", 1),
        ("Wednesday", 2),
        ("Thursday", 3),
        ("Friday", 4),
        ("Saturday", 5),
        ("Sunday", 6),
    ],
)
def test_day_of_week_sort_key(day_of_week: str, expected_index: int):
    assert DayOfTheWeek.sort_key(day_of_week) == expected_index


def test_all_grouper_sort_key():
    grouper_choices = {AllGrouper(): ["All"]}
    g = next(iter(grouper_choices))
    assert sorted(grouper_choices[g], key=(g.sort_key or None)) == ["All"]


def test_value_grouper_sort_key():
    grouper_choices = {ValueGrouper(index_name="some_index_name"): ["b", "c", "a"]}
    g = next(iter(grouper_choices))
    assert sorted(grouper_choices[g], key=(g.sort_key or None)) == ["a", "b", "c"]


@pytest.mark.parametrize(
    "grouper_choices, expected_sorted_choices",
    [
        (
            {TemporalGrouper(temporal_index=Year()): ["2020", "2021", "2019"]},
            ["2019", "2020", "2021"],
        ),
        (
            {TemporalGrouper(temporal_index=Month()): ["March", "January", "February"]},
            ["January", "February", "March"],
        ),
        (
            {
                TemporalGrouper(temporal_index=YearMonth()): [
                    "2020-03",
                    "2020-01",
                    "2020-02",
                ]
            },
            ["2020-01", "2020-02", "2020-03"],
        ),
        (
            {TemporalGrouper(temporal_index=DayOfTheYear()): ["3", "1", "2"]},
            ["1", "2", "3"],
        ),
        (
            {TemporalGrouper(temporal_index=DayOfTheMonth()): ["3", "1", "2"]},
            ["1", "2", "3"],
        ),
        (
            {
                TemporalGrouper(temporal_index=DayOfTheWeek()): [
                    "Wednesday",
                    "Monday",
                    "Tuesday",
                ]
            },
            ["Monday", "Tuesday", "Wednesday"],
        ),
        (
            {
                TemporalGrouper(temporal_index=Date()): [
                    "2020-01-03",
                    "2020-01-01",
                    "2020-01-02",
                ]
            },
            ["2020-01-01", "2020-01-02", "2020-01-03"],
        ),
    ],
)
def test_temporal_grouper_sort_key(
    grouper_choices: dict[AllGrouper | ValueGrouper | TemporalGrouper, list[str]],
    expected_sorted_choices: list[str],
):
    g = next(iter(grouper_choices))
    assert (
        sorted(grouper_choices[g], key=(g.sort_key or None)) == expected_sorted_choices
    )


def test_spatial_grouper():
    spatial_index_name = "A group name"
    sg = SpatialGrouper(spatial_index_name=spatial_index_name)

    assert isinstance(sg, SpatialGrouper)
    assert sg.spatial_index_name == spatial_index_name
    assert sg.index_name == f"SpatialGrouper_{spatial_index_name}"
    assert sg.sort_key is None
    assert not sg.is_resolved
    assert sg.spatial_regions is None
    assert sg.display_name == spatial_index_name


def test_hash_unresolved_spatial_grouper():
    sg1 = SpatialGrouper(spatial_index_name="Feature Group 1")
    sg2 = SpatialGrouper(spatial_index_name="Feature Group 1")
    sg3 = SpatialGrouper(spatial_index_name="Feature Group 2")

    assert hash(sg1) == hash(sg2)
    assert hash(sg1) != hash(sg3)


@pytest.mark.parametrize(
    "directive, expected_type",
    [
        ("%Y", Year),
        ("%B", Month),
        ("%Y-%m", YearMonth),
        ("%j", DayOfTheYear),
        ("%d", DayOfTheMonth),
        ("%A", DayOfTheWeek),
        ("%H", Hour),
        ("%Y-%m-%d", Date),
    ],
)
def test_temporal_grouper_coerces_directive_string(directive, expected_type):
    """Test that strftime directives are coerced to a TemporalGrouper ."""
    tg = TemporalGrouper(temporal_index=directive)
    assert isinstance(tg.temporal_index, expected_type)


def test_temporal_grouper_raises_on_bad_directive():
    """Test that a non-existant strftime directive fails to validate as a TemporalGrouper."""
    bad_directive = "I'm Baaaaaad"
    with pytest.raises(
        ValueError, match=f"Unknown temporal index directive: {bad_directive}"
    ):
        TemporalGrouper(temporal_index=bad_directive)
