from typing import Annotated

from pydantic import Field
from wt_registry import register

from ecoscope.platform.annotations import AnyDataFrame, AnyGeoDataFrame
from ecoscope.platform.indexes import AllGrouper, SpatialGrouper, UserDefinedGroupers

LEG_MEASURE_COLUMNS = ["leg_active_seconds", "leg_member_days", "patrol_elapsed_seconds"]


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
) -> Annotated[AnyDataFrame, Field()]:
    """Replicate each leg once per spatial feature it touches, scaling its measures by track share.
    A spatial grouper whose index level is already on `legs` is skipped, so a second pass with an
    overlapping grouper set does not scale the measures twice.
    """
    if isinstance(groupers, AllGrouper):
        return legs

    out = legs
    for grouper in groupers:
        if not isinstance(grouper, SpatialGrouper) or grouper.index_name not in indexed_traj.index.names:
            continue
        level = grouper.index_name
        if level in out.index.names:
            continue  # already apportioned by an earlier grouper set (idempotent, like the index tasks)
        dist = indexed_traj.reset_index().groupby(["patrol_segment_id", level])["dist_meters"].sum()
        total = dist.groupby(level=0).transform("sum")
        shares = (dist / total)[total > 0].reset_index(name="share")

        merged = out.assign(_pos=range(len(out))).merge(shares, on="patrol_segment_id", how="left")
        merged.index = out.index[merged.pop("_pos")]
        out = merged
        out["share"] = out["share"].fillna(1.0)
        out[level] = out[level].astype(object).where(out[level].notna(), "None")
        for col in LEG_MEASURE_COLUMNS:
            if col in out.columns:
                out[col] = out[col] * out["share"]
        out = out.drop(columns="share")
        out = out.set_index(level, append=True)
    return out
