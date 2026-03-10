from datetime import datetime, timedelta, timezone
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic.json_schema import SkipJsonSchema
from wt_registry import register

from ecoscope.platform.annotations import AdvancedField


class TimezoneInfo(BaseModel):
    label: str
    tzCode: str
    name: str
    utc_offset: Annotated[str, Field(alias="utc")]

    model_config = ConfigDict(populate_by_name=True)

    @property
    def utc_offset_as_datetime_timezone(self):
        sign = 1 if self.utc_offset[0] == "+" else -1
        hours, minutes = map(int, self.utc_offset[1:].split(":"))
        return timezone(sign * timedelta(hours=hours, minutes=minutes))


DEFAULT_TIME_FORMAT = "%d %b %Y %H:%M:%S"
UTC_TIMEZONEINFO = TimezoneInfo(label="UTC", tzCode="UTC", name="UTC", utc_offset="+00:00")


class TimeRange(BaseModel):
    since: datetime
    until: datetime
    timezone: TimezoneInfo
    time_format: str = DEFAULT_TIME_FORMAT

    @model_validator(mode="after")
    def ensure_timezone_awareness(self) -> "TimeRange":
        both_naive = not self.since.tzinfo and not self.until.tzinfo
        both_aware = self.since.tzinfo and self.until.tzinfo
        if not both_naive and not both_aware:
            raise ValueError("Since and until values must both be timezone naive, or both aware")
        if both_naive:
            tz = self.timezone.utc_offset_as_datetime_timezone
            self.since = self.since.replace(tzinfo=tz)
            self.until = self.until.replace(tzinfo=tz)
        return self


@register(description="Choose the period of time to analyze.")
def set_time_range(
    since: Annotated[datetime, Field(description="The start time")],
    until: Annotated[datetime, Field(description="The end time")],
    timezone: Annotated[
        TimezoneInfo | SkipJsonSchema[None],
        Field(default=None),
    ] = None,
    time_format: Annotated[
        str,
        AdvancedField(
            default=DEFAULT_TIME_FORMAT,
            description="The time format",
        ),
    ] = DEFAULT_TIME_FORMAT,
) -> Annotated[TimeRange, Field(description="Time range filter")]:
    if timezone is None:
        # Assume UTC if no timezone is provided
        timezone = UTC_TIMEZONEINFO
    return TimeRange(since=since, until=until, timezone=timezone, time_format=time_format)


@register()
def get_timezone_from_time_range(
    time_range: TimeRange,
) -> Annotated[TimezoneInfo, Field()]:
    """
    Utility function to return the TimezoneInfo object nested in the provided TimeRange in workflow specs
    """
    return time_range.timezone
