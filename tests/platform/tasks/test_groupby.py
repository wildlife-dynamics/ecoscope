import numpy as np
import pandas as pd
import pytest
from ecoscope.platform.indexes import CompositeFilter, IndexName, IndexValue
from ecoscope.platform.tasks.groupby import (
    groupbykey,
    merge_df,
    set_groupers,
    split_groups,
)
from ecoscope.platform.tasks.groupby._groupby import (
    KeyedIterableOfAny,
    ValueGrouper,
    _groupkey_to_composite_filter,
)


def getvalue(key: CompositeFilter, groups: KeyedIterableOfAny) -> pd.DataFrame:
    """Convenience function to get the values for a given key in a KeyedIterable."""
    return next(iter([v for (k, v) in groups if k == key]))


def test_set_groupers():
    input_groupers = [
        ValueGrouper(index_name="class"),
        ValueGrouper(index_name="order"),
    ]
    output_groupers = set_groupers(input_groupers)
    assert input_groupers == output_groupers


@pytest.mark.parametrize(
    "groupers, index_values, expected",
    [
        (
            ["class", "order"],
            ("bird", "Falconiformes"),
            (("class", "=", "bird"), ("order", "=", "Falconiformes")),
        ),
        (
            ["month", "year"],
            ("jan", "2022"),
            (("month", "=", "jan"), ("year", "=", "2022")),
        ),
    ],
)
def test__groupkey_to_composite_filter(
    groupers: list[IndexName],
    index_values: tuple[IndexValue, ...],
    expected: CompositeFilter,
):
    composite_filter = _groupkey_to_composite_filter(groupers, index_values)
    assert composite_filter == expected


def test_split_groups():
    df = pd.DataFrame(
        [
            ("bird", "Falconiformes", 389.0),
            ("bird", "Psittaciformes", 24.0),
            ("mammal", "Carnivora", 80.2),
            ("mammal", "Primates", np.nan),
            ("mammal", "Carnivora", 58),
            (None, None, 13),
        ],
        index=["falcon", "parrot", "lion", "monkey", "leopard", "sasquatch"],
        columns=("class", "order", "max_speed"),
    )
    groupers = [ValueGrouper(index_name="class"), ValueGrouper(index_name="order")]
    groups = split_groups(df, groupers=groupers)
    assert len(groups) == 5
    assert [k for (k, _) in groups] == [
        (("class", "=", "None"), ("order", "=", "None")),
        (("class", "=", "bird"), ("order", "=", "Falconiformes")),
        (("class", "=", "bird"), ("order", "=", "Psittaciformes")),
        (("class", "=", "mammal"), ("order", "=", "Carnivora")),
        (("class", "=", "mammal"), ("order", "=", "Primates")),
    ]
    assert all(isinstance(group, pd.DataFrame) for group in [v for (_, v) in groups])

    class_bird_order_falconiformes_expected_df = pd.DataFrame(
        [("bird", "Falconiformes", 389.0)],
        index=["falcon"],
        columns=("class", "order", "max_speed"),
    )
    pd.testing.assert_frame_equal(
        getvalue((("class", "=", "bird"), ("order", "=", "Falconiformes")), groups),
        class_bird_order_falconiformes_expected_df,
    )
    class_bird_order_psittaciformes_expected_df = pd.DataFrame(
        [("bird", "Psittaciformes", 24.0)],
        index=["parrot"],
        columns=("class", "order", "max_speed"),
    )
    pd.testing.assert_frame_equal(
        getvalue((("class", "=", "bird"), ("order", "=", "Psittaciformes")), groups),
        class_bird_order_psittaciformes_expected_df,
    )


def test_groupbykey():
    falcon = pd.DataFrame([("falcon", 389.0)], index=["falcon"])
    parrot = pd.DataFrame([("parrot", 24.0)], index=["parrot"])
    lion = pd.DataFrame([("lion", 80.2)], index=["lion"])
    monkey = pd.DataFrame([("monkey", np.nan)], index=["monkey"])

    iterable_0 = [
        ((("class", "=", "bird"),), falcon),
        ((("class", "=", "mammal"),), lion),
    ]
    iterable_1 = [
        ((("class", "=", "bird"),), parrot),
        ((("class", "=", "mammal"),), monkey),
    ]
    output = groupbykey(iterables=[iterable_0, iterable_1])

    # bird and mammal groups are combined
    assert output == [
        ((("class", "=", "bird"),), [falcon, parrot]),
        ((("class", "=", "mammal"),), [lion, monkey]),
    ]


def test_merge_df():
    iterables = [
        (
            (("month", "=", "january"), ("year", "=", "2022")),
            pd.DataFrame({"A": [1]}),
        ),
        (
            (("month", "=", "february"), ("year", "=", "2022")),
            pd.DataFrame({"A": [2]}),
        ),
        (
            (("month", "=", "january"), ("year", "=", "2023")),
            pd.DataFrame({"A": [10]}),
        ),
    ]
    result_df = merge_df(iterables)
    multi_idx = pd.MultiIndex.from_tuples(
        [("january", "2022"), ("february", "2022"), ("january", "2023")],
        names=["month", "year"],
    )
    expected_df = pd.DataFrame({"A": [1, 2, 10]}, index=multi_idx)
    pd.testing.assert_frame_equal(result_df, expected_df)


def test_split_groups_value_grouper_in_index():
    df = pd.DataFrame(
        [
            ("bird", "Falconiformes", 389.0),
            ("bird", "Psittaciformes", 24.0),
            ("mammal", "Carnivora", 80.2),
            ("mammal", "Primates", np.nan),
            ("mammal", "Carnivora", 58),
        ],
        index=pd.Index(data=["falcon", "parrot", "lion", None, "leopard"], name="common_name"),
        columns=("class", "order", "max_speed"),
    )
    groupers = [ValueGrouper(index_name="common_name")]
    groups = split_groups(df, groupers=groupers)
    assert len(groups) == 5
    assert [k for (k, _) in groups] == [
        (("common_name", "=", "None"),),
        (("common_name", "=", "falcon"),),
        (("common_name", "=", "leopard"),),
        (("common_name", "=", "lion"),),
        (("common_name", "=", "parrot"),),
    ]
    assert all(isinstance(group, pd.DataFrame) for group in [v for (_, v) in groups])

    common_name_falcon_expected_df = pd.DataFrame(
        [("bird", "Falconiformes", 389.0)],
        index=pd.Index(["falcon"], name="common_name"),
        columns=("class", "order", "max_speed"),
    )
    pd.testing.assert_frame_equal(
        getvalue((("common_name", "=", "falcon"),), groups),
        common_name_falcon_expected_df,
    )

    common_name_none_expected_df = pd.DataFrame(
        [("mammal", "Primates", np.nan)],
        index=pd.Index(["None"], name="common_name"),
        columns=("class", "order", "max_speed"),
    )
    pd.testing.assert_frame_equal(
        getvalue((("common_name", "=", "None"),), groups),
        common_name_none_expected_df,
    )


def test_split_groups_value_grouper_multi_index():
    index_arrays = [
        ["bird", "bird", "mammal", "mammal", "mammal"],
        ["Falconiformes", "Psittaciformes", "Carnivora", "Primates", None],
    ]
    multi_index = pd.MultiIndex.from_arrays(index_arrays, names=("class", "order"))

    df = pd.DataFrame(
        data={"max_speed": [389.0, 24.0, 80.2, np.nan, 58.0]},
        index=multi_index,
    )
    groupers = [ValueGrouper(index_name="order")]
    groups = split_groups(df, groupers=groupers)
    assert len(groups) == 5
    assert [k for (k, _) in groups] == [
        (("order", "=", "Carnivora"),),
        (("order", "=", "Falconiformes"),),
        (("order", "=", "None"),),
        (("order", "=", "Primates"),),
        (("order", "=", "Psittaciformes"),),
    ]
    assert all(isinstance(group, pd.DataFrame) for group in [v for (_, v) in groups])

    carnivora_expected_df = pd.DataFrame(
        data={"max_speed": [80.2]},
        index=pd.MultiIndex.from_arrays([["mammal"], ["Carnivora"]], names=("class", "order")),
    )
    pd.testing.assert_frame_equal(
        getvalue((("order", "=", "Carnivora"),), groups),
        carnivora_expected_df,
    )

    order_none_expected_df = pd.DataFrame(
        data={"max_speed": [58.0]},
        index=pd.MultiIndex.from_arrays([["mammal"], ["None"]], names=("class", "order")),
    )
    pd.testing.assert_frame_equal(
        getvalue((("order", "=", "None"),), groups),
        order_none_expected_df,
    )
