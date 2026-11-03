from typing import Annotated, cast

import pandas as pd
from pydantic import Field
from wt_registry import register

from ecoscope.platform.annotations import AnyGeoDataFrame


@register()
def drop_inactive_grid_cells(
    df: Annotated[
        AnyGeoDataFrame,
        Field(
            description=(
                "Encounter rate grid (from calculate_encounter_rate_grid). Cells "
                "with no events AND no patrol effort are dropped so they do not "
                "render on the map."
            ),
            exclude=True,
        ),
    ],
    event_column: Annotated[
        str,
        Field(description="Column containing event counts per cell."),
    ] = "event_count",
    patrol_effort_column: Annotated[
        str,
        Field(description="Column containing patrol effort (km) per cell."),
    ] = "patrol_effort_km",
) -> AnyGeoDataFrame:
    """Drop grid cells with both zero events and zero patrol effort.

    If either column is missing, the grid is treated as fully inactive — this
    matches the short-circuit return from ``calculate_encounter_rate_grid``
    when one of the feature layers was skipped, where neither count nor effort
    is available.
    """
    if event_column not in df.columns or patrol_effort_column not in df.columns:
        return cast(AnyGeoDataFrame, df.iloc[0:0].copy())
    active = (df[event_column] > 0) | (df[patrol_effort_column] > 0)
    return cast(AnyGeoDataFrame, df.loc[active].copy())


@register()
def mask_low_effort_cells(
    df: Annotated[
        AnyGeoDataFrame,
        Field(
            description=(
                "Encounter rate grid with classification and colormap already "
                "applied. Cells with patrol effort below the threshold have their "
                "color and bin label overridden so that low-coverage areas render "
                "as a single grey legend category."
            ),
            exclude=True,
        ),
    ],
    patrol_effort_column: Annotated[
        str,
        Field(description="Column containing patrol effort (km) per cell."),
    ] = "patrol_effort_km",
    color_column: Annotated[
        str,
        Field(description="Color column (RGBA arrays) to override for masked cells."),
    ] = "rate_colormap",
    label_column: Annotated[
        str,
        Field(description="Legend label column to override for masked cells."),
    ] = "rate_bins",
    threshold_km: Annotated[
        float,
        Field(description="Patrol effort threshold (km); cells below it are masked."),
    ] = 0.2,
    grey_color: Annotated[
        tuple[int, int, int, int],
        Field(description="RGBA color (0-255) applied to masked cells."),
    ] = (128, 128, 128, 255),
    grey_label: Annotated[
        str,
        Field(description="Legend label applied to masked cells."),
    ] = "< 200 m patrol effort",
) -> AnyGeoDataFrame:
    """Override color and label for grid cells with patrol effort below a threshold.

    Masked rows are moved to the end of the returned frame so the grey legend
    entry first-appears after every colored bin — downstream layer code keeps
    the first occurrence of each unique label, so row order controls legend
    order.
    """
    result = df.copy()
    mask = result[patrol_effort_column] < threshold_km
    if mask.any():
        grey_rgba = tuple(int(c) for c in grey_color)
        for idx in result.index[mask]:
            result.at[idx, color_column] = grey_rgba  # type: ignore[assignment]
        result.loc[mask, label_column] = grey_label
        result = pd.concat([result.loc[~mask], result.loc[mask]])  # type: ignore[assignment]
    return cast(AnyGeoDataFrame, result)
