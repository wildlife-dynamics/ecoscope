import logging
from typing import Annotated, cast

from ecoscope.platform.annotations import AdvancedField, AnyDataFrame
from pydantic import Field
from wt_registry import register

logger = logging.getLogger(__name__)


@register()
def normalize_json_column(
    df: AnyDataFrame,
    column: Annotated[str, Field(description="The column name.")],
    skip_if_not_exists: Annotated[
        bool,
        AdvancedField(description="Skip if the column does not exist.", default=True),
    ] = True,
    sort_columns: Annotated[
        bool,
        AdvancedField(description="Sort new columns alphabetically.", default=True),
    ] = True,
) -> AnyDataFrame:
    import ecoscope  # type: ignore[import-untyped]

    if skip_if_not_exists and column not in df.columns:
        logger.warning("Column '%s' does not exist in DataFrame. Skipping normalization.", column)
    else:
        ecoscope.io.earthranger_utils.normalize_column(df, column, sort_columns)

    return cast(
        AnyDataFrame,
        df,
    )


@register()
def normalize_numeric_column(
    df: AnyDataFrame,
    column: Annotated[str, Field(description="The column to normalize, values must be numeric.")],
    output_column_name: Annotated[
        str | None,
        Field(description="If provided, normalized values will be added as a new column."),
    ],
) -> AnyDataFrame:
    from pandas.api.types import is_numeric_dtype

    if not is_numeric_dtype(df[column]):
        raise ValueError(f"Provided column {column} must contain only numeric values")

    normalized_values = (df[column] - df[column].mean()) / df[column].std()
    df[output_column_name if output_column_name else column] = normalized_values

    return cast(
        AnyDataFrame,
        df,
    )
