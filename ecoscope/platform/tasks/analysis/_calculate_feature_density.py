from typing import Annotated, Literal

from pydantic import Field
from pydantic.json_schema import SkipJsonSchema
from wt_registry import register

from ecoscope.platform.annotations import AnyGeoDataFrame


@register()
def calculate_feature_density(
    geodataframe: Annotated[
        AnyGeoDataFrame,
        Field(description="The feature data to count or sum per grid cell.", exclude=True),
    ],
    meshgrid: Annotated[
        AnyGeoDataFrame,
        Field(
            description="The grid cells used to aggregate the feature data.",
            exclude=True,
        ),
    ],
    geometry_type: Annotated[
        Literal["point", "line"],
        Field(description="The geometry type of the provided geodataframe"),
    ],
    sum_column: Annotated[
        str | SkipJsonSchema[None],
        Field(
            description=(
                "Sum values in this column per grid cell, rather than counting rows."
                " Leave empty to count rows per cell."
            )
        ),
    ] = None,
) -> AnyGeoDataFrame:
    """
    Count features or sum column values per grid cell.

    Sum mode is optional: when `sum_column` is None or an empty string, rows are
    counted. Sum-column values are coerced to numeric (non-castable values become
    NaN and are ignored by the per-cell sum).
    """
    import pandas as pd

    from ecoscope.analysis.feature_density import (
        calculate_feature_density,
    )

    if not sum_column:
        sum_column = None
    elif sum_column not in geodataframe.columns:
        raise ValueError(
            f"Column '{sum_column}' not found in the feature data."
            f" Available columns: {', '.join(str(c) for c in geodataframe.columns)}"
        )
    else:
        geodataframe = geodataframe.copy()
        geodataframe[sum_column] = pd.to_numeric(geodataframe[sum_column], errors="coerce")

    result = calculate_feature_density(
        selection=geodataframe,
        grid=meshgrid,
        geometry_type=geometry_type,
        sum_column=sum_column,
    )

    return result
