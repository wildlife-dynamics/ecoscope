import logging
from enum import Enum
from typing import Annotated, cast

from pydantic import Field
from wt_registry import register

from ecoscope.platform.annotations import AnyDataFrame

logger = logging.getLogger(__name__)


class ComparisonOperator(Enum):
    EQUAL = "equal"
    GE = "ge"
    GT = "gt"
    LE = "le"
    LT = "lt"
    NE = "ne"

    @classmethod
    def from_string(cls, value: str):
        try:
            return cls(value.lower())
        except ValueError:
            raise ValueError(f"'{value}' is not a valid FieldType. Valid options are: {[e.value for e in cls]}")


@register()
def filter_df(
    df: Annotated[
        AnyDataFrame,
        Field(
            description="The dataframe.",
            exclude=True,
        ),
    ],
    column_name: Annotated[str, Field(description="The column name to filter on.")],
    op: Annotated[ComparisonOperator, Field(description="The comparison operator")],
    value: Annotated[str, Field(description="The comparison operand")],
    reset_index: Annotated[bool, Field(description="If reset index, default is False")] = False,
) -> AnyDataFrame:
    match op:
        case ComparisonOperator.EQUAL:
            result_df = df[df[column_name] == value]
        case ComparisonOperator.GE:
            result_df = df[df[column_name] >= value]
        case ComparisonOperator.GT:
            result_df = df[df[column_name] > value]
        case ComparisonOperator.LE:
            result_df = df[df[column_name] <= value]
        case ComparisonOperator.LT:
            result_df = df[df[column_name] < value]
        case ComparisonOperator.NE:
            result_df = df[df[column_name] != value]

    if reset_index:
        result_df = result_df.reset_index()

    return cast(AnyDataFrame, result_df)
