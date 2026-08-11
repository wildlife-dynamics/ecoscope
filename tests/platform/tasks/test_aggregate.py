import pandas as pd
import pytest

from ecoscope.platform.tasks.analysis import (
    apply_arithmetic_operation,
    dataframe_column_first_unique,
    dataframe_column_max,
    dataframe_column_mean,
    dataframe_column_min,
    dataframe_column_nunique,
    dataframe_column_percentile,
    dataframe_column_sum,
    dataframe_count,
)


def test_count():
    df = pd.DataFrame({"data": [1, 2, 3]})
    result = dataframe_count(df)
    assert result == 3


def test_mean():
    df = pd.DataFrame({"data": [1, 2, 3]})
    result = dataframe_column_mean(df, "data")
    assert result == 2.0


def test_sum():
    df = pd.DataFrame({"data": [1, 2, 3]})
    result = dataframe_column_sum(df, "data")
    assert result == 6


def test_max():
    df = pd.DataFrame({"data": [1, 2, 3]})
    result = dataframe_column_max(df, "data")
    assert result == 3


def test_min():
    df = pd.DataFrame({"data": [1, 2, 3]})
    result = dataframe_column_min(df, "data")
    assert result == 1


def test_nunique():
    df = pd.DataFrame({"data": [1, 2, 3, 1]})
    result = dataframe_column_nunique(df, "data")
    assert result == 3


def test_first_unique():
    df = pd.DataFrame({"data": [1, 2, 3, 1]})
    result = dataframe_column_first_unique(df, "data")
    assert result == 1


def test_percentile():
    df = pd.DataFrame({"data": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})
    result = dataframe_column_percentile(df, "data", 50)
    assert result == 5.5


@pytest.mark.parametrize(
    "a, b, operation, expected",
    [
        (1, 2, "add", 3),
        (3, 2, "subtract", 1),
        (2, 3, "multiply", 6),
        (6, 3, "divide", 2),
        (7, 3, "floor_divide", 2),
        (7, 3, "modulo", 1),
        (2, 3, "power", 8),
        (2, 3, "min", 2),
        (2, 3, "max", 3),
    ],
)
def test_apply_arithmetic_operation(a, b, operation, expected):
    result = apply_arithmetic_operation(a, b, operation)
    assert result == expected


def test_aggregation_models_roundtrip():
    from ecoscope.platform.tasks.analysis._aggregation import (
        CountAggregation,
        SumOfColumnAggregation,
    )

    assert CountAggregation().count_or_sum == "Count"
    assert SumOfColumnAggregation(column="Number of Animals").column == "Number of Animals"


def test_make_aggregation_json_schema_extra_flattens_to_conditional_object():
    from ecoscope.platform.tasks.analysis._aggregation import (
        make_aggregation_json_schema_extra,
    )

    extra = make_aggregation_json_schema_extra(
        aggregation_title="Rate Based On",
        count_title="Number of Events",
        sum_title="Total of an Event Field",
        column_title="Event Field to Total",
        column_description="desc",
    )
    schema = {"anyOf": [{"$ref": "#/$defs/CountAggregation"}], "default": None}
    extra(schema)

    assert "anyOf" not in schema
    assert schema["type"] == "object"
    select = schema["properties"]["count_or_sum"]
    assert select["title"] == "Rate Based On"
    assert [o["title"] for o in select["oneOf"]] == ["Number of Events", "Total of an Event Field"]
    conditional = schema["allOf"][0]
    assert conditional["if"]["properties"]["count_or_sum"]["const"] == "Sum of Column"
    column = conditional["then"]["properties"]["column"]
    assert column["title"] == "Event Field to Total"
    assert "default" not in column
    assert "additionalProperties" not in schema
