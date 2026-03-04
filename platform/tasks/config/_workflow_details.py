from dataclasses import dataclass
from typing import Annotated

from pydantic import Field
from wt_registry import register


@dataclass
class WorkflowDetails:
    name: str
    description: str
    image_url: str = ""


@register(
    description="Add information that will help to differentiate this workflow from another."
)
def set_workflow_details(
    name: Annotated[str, Field(title="Workflow Name")],
    description: Annotated[str, Field(title="Workflow Description", default="")] = "",
    image_url: Annotated[
        str, Field(description="An image url", default="", exclude=True)
    ] = "",  # This is excluded due to https://github.com/wildlife-dynamics/compose/issues/308
) -> WorkflowDetails:
    return WorkflowDetails(
        name=name,
        description=description,
        image_url=image_url,
    )
