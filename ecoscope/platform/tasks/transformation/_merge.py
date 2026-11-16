from typing import Annotated, List, Literal, cast

import pandas as pd
from pydantic import Field
from wt_registry import register

from ecoscope.platform.annotations import AnyDataFrame


@register()
def merge_two_dataframes(
    left: Annotated[AnyDataFrame, Field(description="The left DataFrame.")],
    right: Annotated[AnyDataFrame, Field(description="The right DataFrame.")],
    how: Annotated[
        Literal["left", "right", "outer", "inner", "cross"],
        Field(description="Type of merge to be performed."),
    ] = "inner",
    on: Annotated[
        str | List[str] | None,
        Field(description="Column or index level names to join on."),
    ] = None,
    left_on: Annotated[
        str | List[str] | None,
        Field(description="Column or index level names to join on in the left DataFrame."),
    ] = None,
    right_on: Annotated[
        str | List[str] | None,
        Field(description="Column or index level names to join on in the right DataFrame."),
    ] = None,
    left_index: Annotated[
        bool,
        Field(description="Use the index from the left DataFrame as the join key(s)."),
    ] = False,
    right_index: Annotated[
        bool,
        Field(description="Use the index from the right DataFrame as the join key(s)."),
    ] = False,
    fillna_value: Annotated[
        float | int | str | None,
        Field(description="Value to use for filling null values arising from the merge."),
    ] = None,
) -> AnyDataFrame:
    """Merge two DataFrames with `pandas.merge`, optionally filling nulls the merge introduces.

    On a left (right) join with differently named keys, the right (left) key column
    duplicates the kept one and is dropped: `left_on="id", right_on="user_id"` keeps `id` only.
    """
    merged_df = pd.merge(
        left=left,
        right=right,
        how=how,
        on=on,
        left_on=left_on,
        right_on=right_on,
        left_index=left_index,
        right_index=right_index,
    )

    if drop_col := {"left": right_on, "right": left_on}.get(how):
        merged_df = merged_df.drop(
            columns=[drop_col] if isinstance(drop_col, str) else drop_col,
            errors="ignore",
        )

    if fillna_value is not None:
        merged_df = merged_df.fillna(fillna_value)

    return cast(AnyDataFrame, merged_df)
