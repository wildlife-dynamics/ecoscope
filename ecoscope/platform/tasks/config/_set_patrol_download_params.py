from typing import Annotated

from pydantic import Field
from pydantic.json_schema import SkipJsonSchema
from wt_registry import register

from ecoscope.platform.annotations import AdvancedField  # type: ignore[import-untyped]
from ecoscope.platform.tasks.io._persist import (  # type: ignore[import-untyped]
    ResultsFileType as FileType,
)

PatrolDownloadParams = tuple[list[FileType], str, list[FileType], str]


@register()
def set_patrol_download_params(
    track_filetypes: Annotated[
        list[FileType] | SkipJsonSchema[None],
        Field(
            title="Patrol Track Filetype(s)",
            description="The output format for the patrol track files.",
            json_schema_extra={"uniqueItems": True},
        ),
    ] = None,
    event_filetypes: Annotated[
        list[FileType] | SkipJsonSchema[None],
        Field(
            title="Patrol Event Filetype(s)",
            description="The output format for the patrol event files.",
            json_schema_extra={"uniqueItems": True},
        ),
    ] = None,
    track_filename_prefix: Annotated[
        str,
        AdvancedField(
            default="patrol_tracks",
            title="Patrol Track Filename prefix",
            description=(
                "A name added to the beginning of your patrol track output files. "
                "A unique ID is automatically added to the end to prevent duplicates."
            ),
        ),
    ] = "patrol_tracks",
    event_filename_prefix: Annotated[
        str,
        AdvancedField(
            default="patrol_events",
            title="Patrol Event Filename prefix",
            description=(
                "A name added to the beginning of your patrol event output files. "
                "A unique ID is automatically added to the end to prevent duplicates."
            ),
        ),
    ] = "patrol_events",
) -> PatrolDownloadParams:
    if track_filetypes is None:
        track_filetypes = ["parquet"]
    if event_filetypes is None:
        event_filetypes = ["parquet"]
    return (
        track_filetypes,
        track_filename_prefix,
        event_filetypes,
        event_filename_prefix,
    )


@register()
def get_patrol_track_filetypes(
    params: Annotated[PatrolDownloadParams, Field(title="")],
) -> list[FileType]:
    return params[0]


@register()
def get_patrol_track_filename_prefix(
    params: Annotated[PatrolDownloadParams, Field(title="")],
) -> str:
    return params[1]


@register()
def get_patrol_event_filetypes(
    params: Annotated[PatrolDownloadParams, Field(title="")],
) -> list[FileType]:
    return params[2]


@register()
def get_patrol_event_filename_prefix(
    params: Annotated[PatrolDownloadParams, Field(title="")],
) -> str:
    return params[3]
