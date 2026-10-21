import logging
from typing import Annotated, cast

from pydantic import Field
from wt_registry import register

from ecoscope.platform.annotations import AnyDataFrame
from ecoscope.platform.indexes import AllGrouper, UserDefinedGroupers, ValueGrouper

logger = logging.getLogger(__name__)


@register()
def explode(
    df: AnyDataFrame,
    column_name: Annotated[str, Field(description="The column name to explode.")],
    ignore_index: Annotated[bool, Field(description="Whether to ignore the index.")],
) -> AnyDataFrame:
    return cast(
        AnyDataFrame,
        df.explode(column_name, ignore_index),
    )


@register()
def explode_for_groupers(
    df: AnyDataFrame,
    groupers: Annotated[
        AllGrouper | UserDefinedGroupers,
        Field(description="The selected groupers; list-valued columns they group by are exploded."),
    ],
) -> AnyDataFrame:
    if isinstance(groupers, AllGrouper):
        return df
    for grouper in groupers:
        col = grouper.index_name if isinstance(grouper, ValueGrouper) else None
        if col in df.columns and df[col].dropna().map(lambda v: isinstance(v, list)).any():
            df = df.explode(col)
            df[col] = df[col].where(df[col].notna(), "None")
    return df
