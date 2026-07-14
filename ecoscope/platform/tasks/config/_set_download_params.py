from typing import Annotated

from pydantic import Field
from pydantic.json_schema import SkipJsonSchema
from wt_registry import register

from ecoscope.platform.annotations import AdvancedField  # type: ignore[import-untyped]
from ecoscope.platform.tasks.io._persist import (  # type: ignore[import-untyped]
    ResultsFileType as FileType,
)

DownloadParams = tuple[list[FileType], str, list[FileType], str, bool]


@register()
def set_download_params(
    track_filetypes: Annotated[
        list[FileType] | SkipJsonSchema[None],
        Field(
            title="Track Filetypes",
            description="The output format for the subject track files.",
            json_schema_extra={"uniqueItems": True},
        ),
    ] = None,
    download_gps_points: Annotated[
        bool,
        # No description: checkbox helper text is supplied below the box via
        # uiSchema ui:help in the consuming workflow (description renders above).
        Field(title="Download GPS points"),
    ] = False,
    gps_point_filetypes: Annotated[
        list[FileType] | SkipJsonSchema[None],
        Field(
            title="GPS Point Filetypes",
            description="The output format for the GPS point (relocation) files.",
            json_schema_extra={"uniqueItems": True},
        ),
    ] = None,
    track_filename_prefix: Annotated[
        str,
        AdvancedField(
            default="subject_tracks",
            title="Track filename prefix",
            description=(
                "A name added to the beginning of your track output files. "
                "A unique ID is automatically added to the end to prevent duplicates."
            ),
        ),
    ] = "subject_tracks",
    gps_point_filename_prefix: Annotated[
        str,
        AdvancedField(
            default="relocations",
            title="GPS point filename prefix",
            description=(
                "A name added to the beginning of your GPS point output files. "
                "A unique ID is automatically added to the end to prevent duplicates."
            ),
        ),
    ] = "relocations",
) -> DownloadParams:
    if track_filetypes is None:
        track_filetypes = ["parquet"]
    if gps_point_filetypes is None:
        gps_point_filetypes = ["parquet"]
    return (
        track_filetypes,
        track_filename_prefix,
        gps_point_filetypes,
        gps_point_filename_prefix,
        download_gps_points,
    )


@register()
def get_track_filetypes(
    params: Annotated[DownloadParams, Field(title="")],
) -> list[FileType]:
    return params[0]


@register()
def get_track_filename_prefix(
    params: Annotated[DownloadParams, Field(title="")],
) -> str:
    return params[1]


@register()
def get_gps_point_filetypes(
    params: Annotated[DownloadParams, Field(title="")],
) -> list[FileType]:
    return params[2]


@register()
def get_gps_point_filename_prefix(
    params: Annotated[DownloadParams, Field(title="")],
) -> str:
    return params[3]


@register()
def get_skip_relocation_persist(
    params: Annotated[DownloadParams, Field(title="")],
) -> bool:
    # download_gps_points is the positive toggle; maybe_skip_df wants the negation.
    return not params[4]
