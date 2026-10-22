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
