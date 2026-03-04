from copy import deepcopy
from dataclasses import dataclass
from typing import Annotated, Any, TypeAlias

import pandas as pd
import pandera.pandas as pa
import pandera.typing as pa_typing
from ecoscope.platform.annotations import (
    AnyGeoDataFrame,
    DataFrame,
    EmptyDataFrame,
    GeoDataFrameBaseSchema,
    JsonSerializableDataFrameModel,
    StrictGeoDataFrameBaseSchema,
)
from pydantic import AfterValidator


def _validate_df_columns_are_timezone_aware(df: pd.DataFrame, columns: list[str]):
    """
    Validate that the given df columns are timezone aware timestamps,
    while remaining agnostic of the specific timezone.
    """
    return all(
        # First check we actually have timestamps via the column dtype
        # Then check that all timestamps in the column are timezone aware
        pd.api.types.is_datetime64_ns_dtype(df[col])
        and df[col].apply(lambda x: x.tzinfo is not None).all()
        for col in columns
    )


def _validate_df_has_required_attrs(df: pd.DataFrame, required: set[str]):
    """
    Validate that the given df contains the required keys in its attr field
    """
    return required.issubset(df.attrs)


class RelocationsGDFSchema(StrictGeoDataFrameBaseSchema):
    groupby_col: pa_typing.Series[str] = pa.Field()
    fixtime: pa_typing.Series[Any] = pa.Field()
    junk_status: pa_typing.Series[bool] = pa.Field()

    @pa.dataframe_check
    def check_timezone_aware_columns(cls, df: pd.DataFrame):  # type: ignore[misc]
        return _validate_df_columns_are_timezone_aware(df, ["fixtime"])


class EventGDFSchema(GeoDataFrameBaseSchema):
    event_type: pa_typing.Series[str] = pa.Field()
    time: pa_typing.Series[Any] = pa.Field()

    @pa.dataframe_check
    def check_timezone_aware_columns(cls, df: pd.DataFrame):  # type: ignore[misc]
        return _validate_df_columns_are_timezone_aware(df, ["time"])


class EventsWithDisplayNamesGDFSchema(EventGDFSchema):
    event_type_display: pa_typing.Series[str] = pa.Field()


def _add_missing_column_data(
    df: pd.DataFrame, columns_with_defaults: dict[str, str | dict]
):
    for column_name, default_value in columns_with_defaults.items():
        if column_name not in df.columns:
            df[column_name] = None

        if isinstance(default_value, dict):
            # Since pandas assigns dicts via a mapping and here we want to assign the literal dict
            df.loc[df[column_name].isna(), column_name] = [
                deepcopy(default_value)
                for _ in range(len(df.loc[df[column_name].isna(), column_name]))
            ]
        else:
            df.fillna({column_name: default_value}, inplace=True)

    return df


def _subject_group_obs_optional_columns(df: pd.DataFrame):
    return _add_missing_column_data(
        df,
        {
            "extra__subject__name": "None",
            "extra__subject__subject_subtype": "None",
            "extra__subject__sex": "None",
        },
    )


def _patrol_obs_optional_columns(df: pd.DataFrame):
    return _add_missing_column_data(
        df,
        {
            "patrol_type__value": "None",
            "patrol_serial_number": "None",
            "patrol_status": "None",
            "patrol_subject": "None",
        },
    )


def _patrol_obs_optional_columns_coerce_patrol_serial(df: pd.DataFrame):
    # Patrol Serial can be numeric, so coerce to a string here
    df.patrol_serial_number = df.patrol_serial_number.apply(
        lambda x: str(int(x) if isinstance(x, float) else str(x))
    )
    return df


def _events_optional_columns(df: pd.DataFrame):
    return _add_missing_column_data(
        df,
        {
            "event_category": "None",
            "reported_by": {"name": "None"},
        },
    )


class StrictPatrolObservationsGDFSchema(RelocationsGDFSchema):
    patrol_type__value: pa_typing.Series[str] = pa.Field()
    patrol_serial_number: pa_typing.Series[str] = pa.Field()
    patrol_status: pa_typing.Series[str] = pa.Field()
    patrol_subject: pa_typing.Series[str] = pa.Field()


class StrictSubjectGroupObservationsGDFSchema(RelocationsGDFSchema):
    extra__subject__name: pa_typing.Series[str] = pa.Field()
    extra__subject__subject_subtype: pa_typing.Series[str] = pa.Field()
    extra__subject__sex: pa_typing.Series[str] = pa.Field()


class StrictEventsGDFSchema(EventGDFSchema):
    event_category: pa_typing.Series[str] = pa.Field()
    reported_by: pa_typing.Series[Any] = (
        pa.Field()
    )  # TODO https://github.com/wildlife-dynamics/ecoscope-workflows/issues/1095


def _patrol_obs_strict(df: pd.DataFrame):
    return StrictPatrolObservationsGDFSchema.validate(df)


def _subject_group_obs_strict(df: pd.DataFrame):
    return StrictSubjectGroupObservationsGDFSchema.validate(df)


def _events_strict(df: pd.DataFrame):
    return StrictEventsGDFSchema.validate(df)


SubjectGroupObservationsGDF: TypeAlias = Annotated[
    DataFrame[RelocationsGDFSchema],
    AfterValidator(_subject_group_obs_optional_columns),
    AfterValidator(_subject_group_obs_strict),
]
PatrolObservationsGDF: TypeAlias = Annotated[
    DataFrame[RelocationsGDFSchema],
    AfterValidator(_patrol_obs_optional_columns),
    AfterValidator(_patrol_obs_optional_columns_coerce_patrol_serial),
    AfterValidator(_patrol_obs_strict),
]
EventGDF: TypeAlias = Annotated[
    DataFrame[EventGDFSchema],
    AfterValidator(_events_optional_columns),
    AfterValidator(_events_strict),
]
EventsWithDisplayNamesGDF: TypeAlias = Annotated[
    DataFrame[EventsWithDisplayNamesGDFSchema],
    AfterValidator(_events_optional_columns),
    AfterValidator(_events_strict),
]


class TrajectoryGDFSchema(StrictGeoDataFrameBaseSchema):
    groupby_col: pa_typing.Series[str] = pa.Field()
    segment_start: pa_typing.Series[Any] = pa.Field()
    segment_end: pa_typing.Series[Any] = pa.Field()
    timespan_seconds: pa_typing.Series[float] = pa.Field()
    dist_meters: pa_typing.Series[float] = pa.Field()
    speed_kmhr: pa_typing.Series[float] = pa.Field()
    heading: pa_typing.Series[float] = pa.Field()
    junk_status: pa_typing.Series[bool] = pa.Field()

    @pa.dataframe_check
    def check_timezone_aware_columns(cls, df: pd.DataFrame):  # type: ignore[misc]
        return _validate_df_columns_are_timezone_aware(
            df, ["segment_start", "segment_end"]
        )


class PatrolsDFSchema(JsonSerializableDataFrameModel):
    id: pa_typing.Series[str] = pa.Field()
    state: pa_typing.Series[str] = pa.Field()
    serial_number: pa_typing.Series[int] = pa.Field()
    patrol_segments: pa_typing.Series[Any] = (
        pa.Field()
    )  # TODO https://github.com/wildlife-dynamics/ecoscope-workflows/issues/1095


TrajectoryGDF: TypeAlias = Annotated[DataFrame[TrajectoryGDFSchema], ...]
PatrolsDF: TypeAlias = Annotated[DataFrame[PatrolsDFSchema], ...]


class RegionsGDFSchema(GeoDataFrameBaseSchema):
    pk: pa_typing.Series[str] = pa.Field()
    name: pa_typing.Series[str] = pa.Field()
    short_name: pa_typing.Series[str] = pa.Field()
    feature_type: pa_typing.Series[str] = pa.Field()
    metadata: pa_typing.Series[Any] = (
        pa.Field()
    )  # TODO https://github.com/wildlife-dynamics/ecoscope-workflows/issues/1095

    @pa.dataframe_check
    def metadata_has_required_data(cls, df: pd.DataFrame):  # type: ignore[misc]
        required_attrs = {"id", "display_name"}
        return required_attrs.issubset(df["metadata"].iloc[0])


RegionsGDF: TypeAlias = Annotated[DataFrame[RegionsGDFSchema], ...]


@dataclass
class SpatialFeaturesGroup:
    id: str
    name: str
    description: str
    url: str
    feature_count: int
    created_at: str
    updated_at: str
    features: AnyGeoDataFrame | EmptyDataFrame
