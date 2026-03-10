import logging
from typing import Annotated, cast

from pydantic import Field
from wt_registry import register

from ecoscope.platform.annotations import AnyDataFrame

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
