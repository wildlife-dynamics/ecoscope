from typing import Annotated

from pydantic import Field
from pydantic.json_schema import SkipJsonSchema
from wt_registry import register
from wt_task.skip import SkippedDependencyFallback

from ecoscope.platform.annotations import EmptyDataFrame
from ecoscope.platform.schemas import SubjectGroupObservationsGDF
from ecoscope.platform.tasks.skip import skip_gdf_fallback_to_none


@register()
def mixed_subtype_warning(
    subject_obs: Annotated[
        SubjectGroupObservationsGDF | EmptyDataFrame | SkipJsonSchema[None],
        Field(),
        SkippedDependencyFallback(skip_gdf_fallback_to_none),
    ],
) -> str | None:
    """
    Utility function to provide a warning string to a workflow spec in the event of a mixed subject subtype
    """
    warning = "This workflow was run with mixed subtypes"
    return (
        warning
        if subject_obs is not None
        and not subject_obs.empty
        and len(subject_obs["extra__subject__subject_subtype"].unique()) > 1
        else None
    )
