from typing import Annotated, cast

from ecoscope.platform.annotations import AnyDataFrame
from pydantic import Field
from pydantic.json_schema import SkipJsonSchema
from wt_registry import register


@register()
def transpose(
    df: AnyDataFrame,
    transposed_column_name: Annotated[
        str | SkipJsonSchema[None],
        Field(description="If provided, the transposed index will be a column with this name"),
    ] = None,
) -> AnyDataFrame:
    from pandas import RangeIndex

    transposed = df.transpose()
    if transposed_column_name:
        transposed = transposed.reset_index(names=transposed_column_name)

    if isinstance(df.index, RangeIndex):
        transposed = transposed.rename(columns={col: str(col) for col in transposed.columns if isinstance(col, int)})

    return cast(AnyDataFrame, transposed)
