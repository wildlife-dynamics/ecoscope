import logging
from typing import Annotated, cast

import numpy as np
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
    """Explode list-valued columns that the selected groupers group by, so each list
    element gets its own row (full credit per member or asset). Empty entries become
    "None". Returns the frame unchanged for `AllGrouper` or when no grouper column holds lists.

    When exploding `patrol_member_names`, `leg_member_days` (if present) is first divided
    by each row's member count (minimum 1), so each member row carries one member's days.
    """
    if isinstance(groupers, AllGrouper):
        return df
    for grouper in groupers:
        if not isinstance(grouper, ValueGrouper):
            continue
        col = grouper.index_name
        if col in df.columns and df[col].dropna().map(lambda v: isinstance(v, (list, tuple, np.ndarray))).any():
            if col == "patrol_member_names" and "leg_member_days" in df.columns:
                counts = df[col].map(lambda v: len(v) if isinstance(v, (list, tuple, np.ndarray)) else 1).clip(lower=1)
                df = df.assign(leg_member_days=df["leg_member_days"] / counts)
            df = df.explode(col)
            df[col] = df[col].where(df[col].notna(), "None")
    return df
