import numpy as np
import pandas as pd
import pytest

from ecoscope.platform.tasks.analysis._patrol_summary import set_encounter_rate_metrics
from ecoscope.platform.tasks.analysis._summary import (
    RatioSummaryParam,
    StatSummaryParam,
    summarize_df,
)


def test_default_metrics():
    params = set_encounter_rate_metrics()
    assert [p.display_name for p in params] == [
        "Total Events",
        "Total Duration (hrs)",
        "Events per Hour",
        "Total Distance (km)",
        "Events per Km",
    ]


def test_default_metrics_with_aggregate_column():
    params = set_encounter_rate_metrics(aggregation={"count_or_sum": "Sum of Column", "column": "number_of_animals"})
    assert [p.display_name for p in params] == [
        "Total Number of Animals",
        "Total Duration (hrs)",
        "Number of Animals per Hour",
        "Total Distance (km)",
        "Number of Animals per Km",
    ]

    per_hour = params[2]
    assert per_hour.numerator.aggregator == "sum"
    assert per_hour.numerator.column == "number_of_animals"
    assert per_hour.denominator.column == "timespan_seconds"


def test_total_events_ignores_aggregate_column():
    params = set_encounter_rate_metrics(
        aggregation={"count_or_sum": "Sum of Column", "column": "number_of_animals"},
        metrics=({"metric": "total_events"},),
    )
    (total_events,) = params
    assert total_events.display_name == "Total Events"
    assert isinstance(total_events, StatSummaryParam)
    assert total_events.aggregator == "count"
    assert total_events.column == "event_type"


@pytest.mark.parametrize(
    "aggregation",
    [
        None,
        {"count_or_sum": "Count"},
        {"count_or_sum": "Sum of Column", "column": ""},  # sum mode, field not chosen yet
    ],
)
def test_encounter_metrics_fall_back_to_count_when_no_column(aggregation):
    params = set_encounter_rate_metrics(
        aggregation=aggregation,
        metrics=(
            {"metric": "total_encounter"},
            {"metric": "encounters_per_duration"},
            {"metric": "encounters_per_distance"},
            {"metric": "encounters_per_patrol"},
            {"metric": "encounters_per_patrol_day"},
        ),
    )
    assert [p.display_name for p in params] == [
        "Total Events",
        "Events per Hour",
        "Events per Km",
        "Events per Patrol",
        "Events per Patrol Day",
    ]
    assert params[0].aggregator == "count"
    for rate in params[1:]:
        assert rate.numerator.aggregator == "count"
        assert rate.numerator.column == "event_type"


def test_encounter_metrics_sum_when_column_set():
    params = set_encounter_rate_metrics(
        aggregation={"count_or_sum": "Sum of Column", "column": "number_of_animals"},
        metrics=(
            {"metric": "total_encounter"},
            {"metric": "encounters_per_duration"},
            {"metric": "encounters_per_distance"},
            {"metric": "encounters_per_patrol"},
            {"metric": "encounters_per_patrol_day"},
        ),
    )
    assert [p.display_name for p in params] == [
        "Total Number of Animals",
        "Number of Animals per Hour",
        "Number of Animals per Km",
        "Number of Animals per Patrol",
        "Number of Animals per Patrol Day",
    ]

    total = params[0]
    assert total.aggregator == "sum"
    assert total.column == "number_of_animals"

    for rate in params[1:]:
        assert isinstance(rate, RatioSummaryParam)
        assert rate.numerator.aggregator == "sum"
        assert rate.numerator.column == "number_of_animals"


def test_per_metric_aggregate_column_overrides_task_level():
    params = set_encounter_rate_metrics(
        aggregation={"count_or_sum": "Sum of Column", "column": "number_of_animals"},
        metrics=(
            {"metric": "total_encounter", "aggregate_column": "number_of_snares"},
            {"metric": "encounters_per_duration", "aggregate_column": "number_of_snares"},
            {"metric": "encounters_per_distance", "aggregate_column": "number_of_snares"},
        ),
    )
    assert [p.display_name for p in params] == [
        "Total Number of Snares",
        "Number of Snares per Hour",
        "Number of Snares per Km",
    ]
    assert params[0].column == "number_of_snares"
    assert params[1].numerator.column == "number_of_snares"
    assert params[2].numerator.column == "number_of_snares"


def test_per_metric_aggregate_columns_differ_without_task_level():
    params = set_encounter_rate_metrics(
        metrics=(
            {"metric": "total_encounter", "aggregate_column": "number_of_animals"},
            {"metric": "encounters_per_distance", "aggregate_column": "number_of_snares"},
            {"metric": "total_encounter"},  # no per-metric or task-level field: counts
        ),
    )
    assert [p.display_name for p in params] == [
        "Total Number of Animals",
        "Number of Snares per Km",
        "Total Events",
    ]
    assert params[0].aggregator == "sum"
    assert params[1].numerator.column == "number_of_snares"
    assert params[2].aggregator == "count"


def test_blank_per_metric_aggregate_column_falls_back_to_task_level():
    params = set_encounter_rate_metrics(
        aggregation={"count_or_sum": "Sum of Column", "column": "number_of_animals"},
        metrics=(
            {"metric": "total_encounter", "aggregate_column": ""},
            {"metric": "encounters_per_duration", "aggregate_column": "number_of_snares"},
        ),
    )
    assert [p.display_name for p in params] == [
        "Total Number of Animals",
        "Number of Snares per Hour",
    ]
    assert params[0].column == "number_of_animals"
    assert params[1].numerator.column == "number_of_snares"


def test_rate_units_configurable():
    params = set_encounter_rate_metrics(
        metrics=(
            {"metric": "encounters_per_duration", "unit": "d"},
            {"metric": "encounters_per_distance", "unit": "m"},
        )
    )
    per_day, per_meter = params
    assert per_day.display_name == "Events per Day"
    assert per_day.scale == 86400.0
    assert per_meter.display_name == "Events per Meter"
    assert per_meter.scale == 1.0


def test_per_patrol_rates():
    params = set_encounter_rate_metrics(
        metrics=(
            {"metric": "encounters_per_patrol"},
            {"metric": "encounters_per_patrol_day"},
        )
    )
    per_patrol, per_patrol_day = params
    assert per_patrol.display_name == "Events per Patrol"
    assert per_patrol.denominator.aggregator == "nunique"
    assert per_patrol.denominator.column == "patrol_id"
    assert per_patrol_day.display_name == "Events per Patrol Day"
    assert per_patrol_day.denominator.aggregator == "nunique"
    assert per_patrol_day.denominator.column == "segment_start_date"


def test_patrol_presets_available():
    params = set_encounter_rate_metrics(
        metrics=(
            {"metric": "patrol_count"},
            {"metric": "patrol_days"},
            {"metric": "area_covered_merged", "swath_width_meters": 200.0},
            {"metric": "custom", "display_name": "Max Speed", "aggregator": "max", "column": "speed_kmhr"},
        )
    )
    assert [p.display_name for p in params] == ["Patrol Count", "Patrol Days", "Merged Area Covered (km²)", "Max Speed"]


def test_metric_subset_from_dicts_with_units():
    params = set_encounter_rate_metrics(
        metrics=(
            {"metric": "total_distance", "unit": "m"},
            {"metric": "total_duration", "unit": "d"},
            {"metric": "encounters_per_distance"},
        )
    )
    assert [p.display_name for p in params] == ["Total Distance (m)", "Total Duration (days)", "Events per Km"]


def test_unknown_metric_raises():
    with pytest.raises(ValueError):
        set_encounter_rate_metrics(metrics=({"metric": "not_a_metric"},))


@pytest.fixture
def combined_df():
    # Event columns are NaN on segment rows and vice versa.
    nan = np.nan
    return pd.DataFrame(
        {
            "grp": ["A", "A", "A", "A", "B"],
            "patrol_id": ["p1", "p2", nan, nan, "p3"],
            "dist_meters": [2000.0, 3000.0, nan, nan, 1000.0],
            "timespan_seconds": [3600.0, 3600.0, nan, nan, 1800.0],
            "segment_start_date": ["2026-01-01", "2026-01-01", nan, nan, "2026-01-02"],
            "event_type": [nan, nan, "sighting", "sighting", nan],
            "number_of_animals": [nan, nan, 2.0, 3.0, nan],
            "number_of_snares": [nan, nan, 1.0, 0.0, nan],
        }
    )


def test_end_to_end_with_summarize_df(combined_df):
    result = summarize_df(combined_df, set_encounter_rate_metrics(), groupby_cols=["grp"])

    assert result.loc["A", "Total Events"] == 2
    assert result.loc["A", "Total Duration (hrs)"] == 2.0
    assert result.loc["A", "Events per Hour"] == 1.0
    assert result.loc["A", "Total Distance (km)"] == 5.0
    assert result.loc["A", "Events per Km"] == 0.4

    assert result.loc["B", "Total Events"] == 0
    assert result.loc["B", "Events per Hour"] == 0.0


def test_end_to_end_per_patrol_rates(combined_df):
    result = summarize_df(
        combined_df,
        set_encounter_rate_metrics(
            metrics=({"metric": "encounters_per_patrol"}, {"metric": "encounters_per_patrol_day"})
        ),
        groupby_cols=["grp"],
    )
    assert result.loc["A", "Events per Patrol"] == 1.0
    assert result.loc["A", "Events per Patrol Day"] == 2.0


def test_end_to_end_with_aggregate_column(combined_df):
    result = summarize_df(
        combined_df,
        set_encounter_rate_metrics(
            aggregation={"count_or_sum": "Sum of Column", "column": "number_of_animals"},
            metrics=(
                {"metric": "total_events"},
                {"metric": "total_encounter"},
                {"metric": "encounters_per_duration"},
                {"metric": "encounters_per_distance"},
            ),
        ),
        groupby_cols=["grp"],
    )

    assert result.loc["A", "Total Events"] == 2
    assert result.loc["A", "Total Number of Animals"] == 5.0
    assert result.loc["A", "Number of Animals per Hour"] == 2.5
    assert result.loc["A", "Number of Animals per Km"] == 1.0


def test_end_to_end_with_per_metric_aggregate_columns(combined_df):
    result = summarize_df(
        combined_df,
        set_encounter_rate_metrics(
            metrics=(
                {"metric": "total_encounter", "aggregate_column": "number_of_animals"},
                {"metric": "total_encounter", "aggregate_column": "number_of_snares"},
                {"metric": "encounters_per_duration", "aggregate_column": "number_of_snares"},
            ),
        ),
        groupby_cols=["grp"],
    )

    assert result.loc["A", "Total Number of Animals"] == 5.0
    assert result.loc["A", "Total Number of Snares"] == 1.0
    assert result.loc["A", "Number of Snares per Hour"] == 0.5
