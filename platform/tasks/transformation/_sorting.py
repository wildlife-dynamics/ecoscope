import logging
from typing import Annotated, Literal, cast

from ecoscope.platform.annotations import AdvancedField, AnyDataFrame
from pydantic import Field
from wt_registry import register

logger = logging.getLogger(__name__)


@register()
def sort_values(
    df: AnyDataFrame,
    column_name: Annotated[str, Field(description="The column name to sort values by.")],
    ascending: Annotated[bool, Field(description="Sort ascending if true")] = True,
    na_position: Annotated[
        Literal["first", "last"],
        AdvancedField(description="Where to place NaN values in the sort", default="last"),
    ] = "last",
) -> AnyDataFrame:
    return cast(
        AnyDataFrame,
        df.sort_values(by=column_name, ascending=ascending, na_position=na_position),
    )
