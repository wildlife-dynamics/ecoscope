from typing import Annotated, cast

import numpy as np
import pandas as pd
from pydantic import Field
from wt_registry import register

from ecoscope.platform.annotations import AnyGeoDataFrame


@register()
def calculate_encounter_rate_grid(
    feature_layers: Annotated[
        list[AnyGeoDataFrame],
        Field(
            description=(
                "A list of exactly two GeoDataFrames: [events_gdf, trajectories_gdf]. "
                "Events must have point geometry, trajectories must have line geometry. "
                "Typically produced by groupbykey zipping split events and trajectories."
            ),
            exclude=True,
        ),
    ],
    meshgrid: Annotated[
        AnyGeoDataFrame,
        Field(
            description="Grid cells to aggregate into (e.g. from create_meshgrid).",
            exclude=True,
        ),
    ],
    sum_column: Annotated[
        str,
        Field(
            description=(
                "Sum values in this event column per grid cell as the encounter "
                "rate numerator, rather than counting events. Leave empty to "
                "count events per cell."
            )
        ),
    ] = "",
) -> AnyGeoDataFrame:
    """Encounter rate per grid cell: events (or the per-cell sum of `sum_column`) per km of patrol effort.

    Cells with patrol effort but no events rate 0; cells with no effort are NaN. If either
    feature layer was skipped (fewer than two given), every cell's `density` is NaN. The
    result also carries `event_count`, `patrol_effort_km` and, with `sum_column`, `event_sum`.
    """
    import geopandas as gpd  # type: ignore[import-untyped]

    if len(feature_layers) < 2:
        grid = meshgrid.copy()
        grid["density"] = np.nan
        return cast(AnyGeoDataFrame, grid)

    events_gdf, traj_gdf = feature_layers[0], feature_layers[1]

    if sum_column and sum_column not in events_gdf.columns:
        raise ValueError(
            f"Column '{sum_column}' not found in the events data."
            f" Available columns: {', '.join(str(c) for c in events_gdf.columns)}"
        )

    # Create independent grid copies to avoid mutation
    event_grid = gpd.GeoDataFrame(geometry=meshgrid.geometry.copy(), crs=meshgrid.crs)
    effort_grid = gpd.GeoDataFrame(geometry=meshgrid.geometry.copy(), crs=meshgrid.crs)

    # Project inputs to grid CRS
    events_proj = events_gdf.to_crs(event_grid.crs)  # type: ignore[operator]
    traj_proj = traj_gdf.to_crs(effort_grid.crs)  # type: ignore[operator]

    # Count events per cell
    def count_points(cell):
        mask = events_proj.geometry.within(cell)
        return mask.sum()

    event_grid["event_count"] = event_grid.geometry.apply(count_points)

    if sum_column:
        sum_values = pd.to_numeric(events_proj[sum_column], errors="coerce")

        # Sum column values over events per cell (NaN ignored)
        def sum_points(cell):
            mask = events_proj.geometry.within(cell)
            return sum_values[mask].sum()

        event_grid["event_sum"] = event_grid.geometry.apply(sum_points)

    # Sum patrol line length per cell
    def sum_line_length(cell):
        clipped = traj_proj.clip_by_rect(*cell.bounds)
        clipped = clipped[~clipped.is_empty]
        return clipped.geometry.length.sum()

    effort_grid["effort"] = effort_grid.geometry.apply(sum_line_length)

    # Compute encounter rate (events per km)
    result = meshgrid.copy()
    counts = event_grid["event_count"].to_numpy(dtype=float)
    numerator = event_grid["event_sum"].to_numpy(dtype=float) if sum_column else counts
    effort_m = effort_grid["effort"].to_numpy(dtype=float)
    effort_km = effort_m / 1000.0

    with np.errstate(divide="ignore", invalid="ignore"):
        rate = numerator / effort_km

    rate[~np.isfinite(rate)] = np.nan
    # Keep cells with patrol effort but 0 events as 0 (not NaN)
    # Only NaN cells with no patrol effort at all
    rate[(numerator == 0) & (effort_km > 0)] = 0.0

    result["density"] = rate
    result["event_count"] = counts
    if sum_column:
        result["event_sum"] = numerator
    result["patrol_effort_km"] = np.round(effort_km, 2)

    return cast(AnyGeoDataFrame, result)
