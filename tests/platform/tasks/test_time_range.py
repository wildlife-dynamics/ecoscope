from datetime import datetime, timedelta, timezone

import pytest

from ecoscope.platform.tasks.filter._filter import (
    DEFAULT_TIME_FORMAT,
    UTC_TIMEZONEINFO,
    TimeRange,
    TimezoneInfo,
    set_time_range,
)


def test_time_zone_info_model_validation():
    alias = TimezoneInfo(
        label="UTC (UTC+00:00)",
        tzCode="UTC",
        name="(UTC+00:00) UTC",
        utc="+00:00",
    )
    no_alias = TimezoneInfo(
        label="UTC (UTC+00:00)",
        tzCode="UTC",
        name="(UTC+00:00) UTC",
        utc_offset="+00:00",
    )
    assert alias == no_alias


def test_utc_offset_as_datetime_timezone():
    utc = TimezoneInfo(
        label="UTC (UTC+00:00)",
        tzCode="UTC",
        name="(UTC+00:00) UTC",
        utc_offset="+00:00",
    )
    eat = TimezoneInfo(
        label="Africa/Nairobi (UTC+03:00)",
        tzCode="Africa/Nairobi",
        name="(UTC+03:00) Nairobi",
        utc_offset="+03:00",
    )
    pacific = TimezoneInfo(
        label="America/Los_Angeles (UTC-08:00)",
        tzCode="America/Los_Angeles",
        name="(UTC-08:00) Los Angeles, San Diego, San Jose, San Francisco, Seattle",
        utc_offset="-08:00",
    )

    assert utc.utc_offset_as_datetime_timezone == timezone.utc
    assert eat.utc_offset_as_datetime_timezone == timezone(timedelta(seconds=10800))
    assert pacific.utc_offset_as_datetime_timezone == timezone(timedelta(days=-1, seconds=57600))


def test_set_time_range():
    since = datetime.fromisoformat("2015-01-01T00:00:00+00:00")
    until = datetime.fromisoformat("2016-01-01T00:00:00+00:00")
    timezone = TimezoneInfo(
        label="Africa/Nairobi (UTC+03:00)",
        tzCode="Africa/Nairobi",
        name="(UTC+03:00) Nairobi",
        utc_offset="+03:00",
    )
    time_format = "%d %b %Y %H:%M:%S %Z"
    time_range = set_time_range(since=since, until=until, timezone=timezone, time_format=time_format)

    assert isinstance(time_range, TimeRange)
    assert time_range.since == since
    assert time_range.until == until
    assert time_range.timezone == timezone
    assert time_range.time_format == time_format


def test_set_time_range_with_defaults():
    since = datetime.fromisoformat("2015-01-01T00:00:00+00:00")
    until = datetime.fromisoformat("2016-01-01T00:00:00+00:00")
    time_range = set_time_range(since=since, until=until)

    assert isinstance(time_range, TimeRange)
    assert time_range.since == since
    assert time_range.until == until
    assert time_range.timezone == UTC_TIMEZONEINFO
    assert time_range.time_format == DEFAULT_TIME_FORMAT


def test_set_time_range_mixed_awareness_errors():
    aware_since = datetime.fromisoformat("2015-01-01T00:00:00+00:00")
    aware_until = datetime.fromisoformat("2016-01-01T00:00:00+00:00")
    naive_since = datetime.fromisoformat("2015-01-01T00:00:00")
    naive_until = datetime.fromisoformat("2016-01-01T00:00:00")

    with pytest.raises(ValueError):
        set_time_range(since=aware_since, until=naive_until)

    with pytest.raises(ValueError):
        set_time_range(since=naive_since, until=aware_until)


def test_set_time_range_naive_is_coerced():
    since = datetime.fromisoformat("2015-01-01T00:00:00")
    until = datetime.fromisoformat("2016-01-01T00:00:00")
    expected_since = datetime.fromisoformat("2015-01-01T00:00:00+03:00")
    expected_until = datetime.fromisoformat("2016-01-01T00:00:00+03:00")
    timezone = TimezoneInfo(
        label="Africa/Nairobi (UTC+03:00)",
        tzCode="Africa/Nairobi",
        name="(UTC+03:00) Nairobi",
        utc_offset="+03:00",
    )
    time_range = set_time_range(since=since, until=until, timezone=timezone)

    assert isinstance(time_range, TimeRange)
    assert time_range.since == expected_since
    assert time_range.until == expected_until
    assert time_range.timezone == timezone
