from typing import Annotated

import pandas as pd
from pydantic import Field
from wt_registry import register

from ecoscope.platform.annotations import AnyDataFrame, AnyGeoDataFrame
from ecoscope.platform.indexes import AllGrouper, SpatialGrouper, UserDefinedGroupers

LEG_MEASURE_COLUMNS = ["leg_active_seconds", "leg_member_days", "patrol_elapsed_seconds"]


def _spread_patrol_elapsed(legs: pd.DataFrame, level: str, dist: pd.Series, elapsed_by_patrol: pd.Series) -> pd.Series:
    """Distribute each patrol's elapsed seconds over its (leg, feature) rows by the row's share
    of the patrol's total track length, so a feature gets the patrol's elapsed time in
    proportion to how much of the whole patrol's track lies in it. Rows of a patrol without
    any track keep their value (the earliest leg carries the whole, landing in "None").
    """
    patrol_key = legs["patrol_id"] if "patrol_id" in legs.columns else legs["patrol_segment_id"]
    keys = pd.MultiIndex.from_arrays([legs["patrol_segment_id"], legs[level]])
    seg_dist = pd.Series(dist.reindex(keys).to_numpy(), index=legs.index).fillna(0.0)
    patrol_dist = seg_dist.groupby(patrol_key.to_numpy()).transform("sum")
    patrol_elapsed = patrol_key.map(elapsed_by_patrol).to_numpy(dtype=float)
    spread = patrol_elapsed * seg_dist.to_numpy() / patrol_dist.where(patrol_dist > 0, 1.0).to_numpy()
    return pd.Series(spread, index=legs.index).where(
        (patrol_dist > 0).to_numpy(), legs["patrol_elapsed_seconds"].to_numpy()
    )


@register()
def apportion_legs_by_spatial_index(
    legs: Annotated[AnyDataFrame, Field(description="The leg frame, one row per patrol leg.", exclude=True)],
    indexed_traj: Annotated[
        AnyGeoDataFrame,
        Field(
            description="Track segments with a spatial index and a `patrol_segment_id` column.",
            exclude=True,
        ),
    ],
    groupers: Annotated[
        AllGrouper | UserDefinedGroupers,
        Field(description="Specify how the data should be grouped.", exclude=True),
    ],
    drop_unplaced: Annotated[
        bool,
        Field(description='Drop legs whose track matches no spatial feature instead of a whole "None" group.'),
    ] = False,
) -> Annotated[AnyDataFrame, Field()]:
    """Replicate each leg once per spatial feature it touches, scaling its measures by track share.
    `leg_active_seconds` and `leg_member_days` follow the leg's own track share; the patrol-level
    `patrol_elapsed_seconds` is spread over all of the patrol's rows by their share of the whole
    patrol's track, so a feature touched only by a later leg still gets its part of the elapsed
    time. A spatial grouper whose index level is already on `legs` is skipped, so a second pass
    with an overlapping grouper set does not scale the measures twice. When the track frame has
    no `patrol_segment_id` column (fixes recorded before the leg schema), every leg lands whole in
    the "None" group — or, with `drop_unplaced`, is dropped, as is any leg without a track. An
    empty leg frame is returned as is.
    """
    if isinstance(groupers, AllGrouper) or legs.empty:
        return legs

    elapsed_spread = False
    for grouper in groupers:
        if not isinstance(grouper, SpatialGrouper) or grouper.index_name not in indexed_traj.index.names:
            continue
        level = grouper.index_name
        if level in legs.index.names:
            continue  # already apportioned by an earlier grouper set (idempotent, like the index tasks)
        traj = indexed_traj.reset_index()
        dist = None
        if {"patrol_segment_id", "dist_meters"} <= set(traj.columns):
            dist = traj.groupby(["patrol_segment_id", level])["dist_meters"].sum()
            total = dist.groupby(level=0).transform("sum")
            shares = (dist / total)[total > 0].reset_index(name="share")
        else:
            # Tracks without a leg id (observations recorded before the leg schema):
            # no share can be computed, every leg lands whole in the "None" group.
            shares = pd.DataFrame(
                {
                    "patrol_segment_id": pd.Series(dtype=object),
                    level: pd.Series(dtype=object),
                    "share": pd.Series(dtype=float),
                }
            )

        elapsed_by_patrol = None
        if "patrol_elapsed_seconds" in legs.columns and dist is not None and not elapsed_spread:
            patrol_key = legs["patrol_id"] if "patrol_id" in legs.columns else legs["patrol_segment_id"]
            elapsed_by_patrol = legs["patrol_elapsed_seconds"].groupby(patrol_key.to_numpy()).sum(min_count=1)

        merged = legs.assign(_pos=range(len(legs))).merge(shares, on="patrol_segment_id", how="left")
        merged.index = legs.index[merged.pop("_pos")]
        if drop_unplaced:
            merged = merged[merged["share"].notna()]
        legs = merged
        legs["share"] = legs["share"].fillna(1.0)
        legs[level] = legs[level].astype(object).where(legs[level].notna(), "None")
        measure_columns = [col for col in LEG_MEASURE_COLUMNS if col in legs.columns]
        if elapsed_by_patrol is not None and dist is not None:
            legs["patrol_elapsed_seconds"] = _spread_patrol_elapsed(legs, level, dist, elapsed_by_patrol)
            measure_columns.remove("patrol_elapsed_seconds")
            elapsed_spread = True  # later levels scale the spread value by leg share like the rest
        for col in measure_columns:
            legs[col] = legs[col] * legs["share"]
        legs = legs.drop(columns="share")
        legs = legs.set_index(level, append=True)
    return legs
