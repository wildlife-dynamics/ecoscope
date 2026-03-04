from typing import Annotated

from pydantic import Field
from wt_registry import register


@register()
def placeholder_task(
    placeholder_field: Annotated[int, Field(description="A field for testing")],
) -> bool:
    """
    Dummy task for the purposes of testing the task registry.
    """
    return True
