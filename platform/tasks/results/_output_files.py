from typing import Annotated

from ecoscope.platform.indexes import CompositeFilter
from pydantic import BaseModel, Field
from wt_registry import register


class OutputFiles(BaseModel):
    files: list[str | list[tuple[CompositeFilter, str]]]


@register()
def gather_output_files(
    files: Annotated[
        list[str | list[tuple[CompositeFilter, str]]],
        Field(description="The files to gather.", exclude=True),
    ],
) -> OutputFiles:
    """Gather the output files from the tasks.

    Args:
        files: A list of files to gather output files from.

    Returns:
        A list of output files.
    """
    return OutputFiles(files=files)
