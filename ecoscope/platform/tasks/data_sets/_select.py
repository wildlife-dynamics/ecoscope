from typing import Annotated, cast

from pydantic import Field
from wt_registry import register

from ecoscope.platform.annotations import EmptyDataFrame
from ecoscope.platform.schemas import RegionsGDF


@register()
def select_spatial_features_group(
    groups: Annotated[RegionsGDF | EmptyDataFrame, Field(exclude=True)],
    spatial_features_group_name: Annotated[str, Field(description="The name of the group to use")],
) -> RegionsGDF | EmptyDataFrame:
    """Select one spatial features group from the all-groups table.

    Args:
        groups: The spatial feature groups table; each row's metadata carries the group display name.
        spatial_features_group_name: The display name of the group to keep. Empty means no group.

    Returns:
        The rows of the named group, or an empty frame when no name is given.
    """
    if not spatial_features_group_name:
        return cast(EmptyDataFrame, groups.iloc[0:0])
    if groups.empty:
        raise ValueError("Spatial groupers need a Spatial feature groups data set bound to this output.")
    names = groups["metadata"].map(lambda m: m["display_name"])
    if spatial_features_group_name not in set(names):
        raise ValueError(
            f"Spatial features group {spatial_features_group_name!r} not found; "
            f"available: {', '.join(sorted(set(names)))}"
        )
    return cast(RegionsGDF, groups[names == spatial_features_group_name].reset_index(drop=True))
