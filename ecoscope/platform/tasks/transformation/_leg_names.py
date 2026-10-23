from typing import Annotated, cast

import numpy as np
import pandas as pd
from pydantic import Field
from wt_registry import register

from ecoscope.platform.annotations import AnyDataFrame

LEG_NAME_COLUMNS = ["patrol_team_name", "patrol_member_names", "patrol_asset_names"]


@register()
def add_leg_names_to_tracks(
    df: Annotated[AnyDataFrame, Field(description="Track frame carrying a `patrol_segment_id` column.", exclude=True)],
    legs: Annotated[
        AnyDataFrame,
        Field(description="Leg frame with resolved `patrol_team_name`, `patrol_member_names`, `patrol_asset_names`."),
    ],
) -> Annotated[AnyDataFrame, Field(description="The track frame with the three leg name columns added.")]:
    """Copy `patrol_team_name`, `patrol_member_names` and `patrol_asset_names` from the leg frame
    onto the track frame by `patrol_segment_id`, so the names are resolved once (on the legs) and
    both frames group by the same keys. Rows whose segment has no leg, and every row when the
    frame has no `patrol_segment_id` column (no `patrol_schemas` flag), get "None". Row count,
    order and index are preserved; list-valued name columns stay lists.
    """
    df = df.copy()
    if "patrol_segment_id" not in df.columns or legs.empty:
        for col in LEG_NAME_COLUMNS:
            df[col] = "None"
        return cast(AnyDataFrame, df)

    lookup = legs.drop_duplicates("patrol_segment_id").set_index("patrol_segment_id")
    for col in LEG_NAME_COLUMNS:
        if col not in lookup.columns:
            df[col] = "None"
            continue
        values = df["patrol_segment_id"].map(lookup[col])
        df[col] = values.map(lambda v: v if isinstance(v, (list, tuple, np.ndarray)) or not pd.isna(v) else "None")
    return cast(AnyDataFrame, df)


@register()
def add_patrol_type_display_from_legs(
    df: Annotated[
        AnyDataFrame,
        Field(description="Track or event frame whose `patrol_type` holds the type's value (slug).", exclude=True),
    ],
    legs: Annotated[
        AnyDataFrame,
        Field(description="Leg frame with `patrol_type` as the display name and the slug in `patrol_type__value`."),
    ],
) -> Annotated[
    AnyDataFrame,
    Field(description="The frame with `patrol_type` as the display name and the slug in `patrol_type__value`."),
]:
    """Copy the patrol type display names resolved on the leg frame (`add_patrol_type_display_names`)
    onto a track or event frame by slug lookup, so the three row kinds share one `patrol_type`
    value. `patrol_type` becomes the display name and the slug is kept in `patrol_type__value`
    (left as is when that column already exists). Slugs the leg frame does not carry keep the
    slug, which is also what every row gets when the leg frame is empty or lacks the two
    columns. A frame without `patrol_type` is returned unchanged. Row count, order and index
    are preserved.
    """
    if "patrol_type" not in df.columns:
        return df
    df = df.copy()
    lookup: dict = {}
    if not legs.empty and {"patrol_type", "patrol_type__value"} <= set(legs.columns):
        named = legs[["patrol_type__value", "patrol_type"]].dropna().drop_duplicates("patrol_type__value")
        lookup = dict(zip(named["patrol_type__value"], named["patrol_type"]))
    if "patrol_type__value" not in df.columns:
        df["patrol_type__value"] = df["patrol_type"]
    df["patrol_type"] = df["patrol_type"].map(lambda v: lookup.get(v, v))
    return cast(AnyDataFrame, df)
