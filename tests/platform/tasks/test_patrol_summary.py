import pandas as pd
import pytest
from pydantic import ValidationError

from ecoscope.platform.tasks.analysis._patrol_summary import (
    CustomMetric,
    MergedAreaCoveredMetric,
    TotalDistanceMetric,
    TotalDurationMetric,
    set_patrol_summary_metrics,
)
from ecoscope.platform.tasks.analysis._summary import (
    StatSummaryParam,
    join_summary_tables,
    select_leg_metrics,
    select_track_metrics,
    summarize_df,
)
from ecoscope.platform.tasks.transformation._unit import Unit


def test_set_patrol_summary_metrics_defaults():
    params = set_patrol_summary_metrics()
    assert [p.display_name for p in params] == [
        "Patrol Count",
        "Total Distance (km)",
        "Total Duration (hrs)",
        "Patrol Days",
        "Merged Area Covered (km²)",
        "Unmerged Area Covered (km²)",
    ]
    merged, unmerged = params[4], params[5]
    assert merged.aggregator == unmerged.aggregator == "coverage_area"
    assert merged.merged and not unmerged.merged
    assert merged.swath_width_meters == 500.0


def test_set_patrol_summary_metrics_units():
    dist, dur = set_patrol_summary_metrics(
        [
            TotalDistanceMetric(metric="total_distance", unit="m"),
            TotalDurationMetric(metric="total_duration", unit="d"),
        ]
    )
    assert dist.display_name == "Total Distance (m)"
    assert dist.new_unit == Unit.METER
    assert dur.display_name == "Total Duration (days)"
    assert dur.new_unit == Unit.DAY


def test_set_patrol_summary_metrics_custom_numeric():
    (param,) = set_patrol_summary_metrics(
        [
            {
                "metric": "custom",
                "display_name": "Max Speed",
                "aggregator": "max",
                "column": "speed_kmhr",
            }
        ]
    )
    assert isinstance(param, StatSummaryParam)
    assert param.display_name == "Max Speed"
    assert param.aggregator == "max"
    assert param.column == "speed_kmhr"
    assert param.original_unit is None and param.new_unit is None
    assert param.decimal_places == 2


def test_set_patrol_summary_metrics_custom_tally():
    (param,) = set_patrol_summary_metrics(
        [
            {
                "metric": "custom",
                "display_name": "Rangers",
                "aggregator": "nunique",
                "column": "patrol_subject",
            }
        ]
    )
    assert isinstance(param, StatSummaryParam)
    assert param.aggregator == "nunique"
    assert param.column == "patrol_subject"


def test_set_patrol_summary_metrics_custom_unit_conversion():
    (param,) = set_patrol_summary_metrics(
        [
            {
                "metric": "custom",
                "display_name": "Total Distance (km)",
                "aggregator": "sum",
                "column": "dist_meters",
                "convert_units": True,
                "original_unit": "m",
                "new_unit": "km",
            }
        ]
    )
    assert param.original_unit == Unit.METER
    assert param.new_unit == Unit.KILOMETER
    # hand-typed suffix is not duplicated
    assert param.display_name == "Total Distance (km)"


def test_custom_metric_appends_unit_to_display_name():
    dist, dur = set_patrol_summary_metrics(
        [
            {
                "metric": "custom",
                "display_name": "Avg Leg Distance",
                "aggregator": "mean",
                "column": "dist_meters",
                "convert_units": True,
                "original_unit": "m",
                "new_unit": "km",
            },
            {
                "metric": "custom",
                "display_name": "Max Leg Duration",
                "aggregator": "max",
                "column": "timespan_seconds",
                "convert_units": True,
                "original_unit": "s",
                "new_unit": "h",
            },
        ]
    )
    assert dist.display_name == "Avg Leg Distance (km)"
    assert dur.display_name == "Max Leg Duration (hrs)"


def test_custom_metric_units_ignored_when_unchecked():
    metric = CustomMetric(
        metric="custom",
        display_name="Distance",
        aggregator="sum",
        column="dist_meters",
        convert_units=False,
        original_unit=Unit.METER,
        new_unit=Unit.KILOMETER,
    )
    param = metric.to_summary_param()
    assert param.original_unit is None and param.new_unit is None
    # no unit suffix without conversion — the name stays exactly as typed
    assert param.display_name == "Distance"


def test_custom_metric_requires_both_units_when_checked():
    with pytest.raises(ValidationError, match="original and a new unit"):
        CustomMetric(
            metric="custom",
            display_name="Distance",
            aggregator="sum",
            column="dist_meters",
            convert_units=True,
            original_unit=Unit.METER,
        )


def test_custom_metric_schema_hides_units_behind_conditional():
    schema = CustomMetric.model_json_schema()
    assert "original_unit" not in schema["properties"]
    assert "new_unit" not in schema["properties"]
    conditional = schema["allOf"][0]
    assert conditional["if"]["properties"]["convert_units"]["const"] is True
    revealed = conditional["then"]["properties"]
    assert {o["const"] for o in revealed["original_unit"]["oneOf"]} == {u.value for u in Unit}
    assert {o["const"] for o in revealed["new_unit"]["oneOf"]} == {u.value for u in Unit}


def test_total_distance_metric_labeled_units_and_decimals():
    schema = TotalDistanceMetric.model_json_schema()
    unit = schema["properties"]["unit"]
    assert "enum" not in unit
    assert unit["oneOf"] == [
        {"const": "km", "title": "Kilometers (km)"},
        {"const": "m", "title": "Meters (m)"},
    ]
    # presets don't expose decimal places; always 2
    assert "decimal_places" not in schema["properties"]
    param = TotalDistanceMetric(metric="total_distance").to_summary_param()
    assert param.decimal_places == 2


def test_set_patrol_summary_metrics_accepts_dicts():
    params = set_patrol_summary_metrics([{"metric": "area_covered_unmerged", "swath_width_meters": 250.0}])
    assert params[0].display_name == "Unmerged Area Covered (km²)"
    assert params[0].merged is False
    assert params[0].swath_width_meters == 250.0


def test_merged_area_covered_metric_defaults():
    m = MergedAreaCoveredMetric(metric="area_covered_merged")
    assert m.swath_width_meters == 500.0
    assert m.to_summary_param().merged is True


def test_set_patrol_summary_metrics_defaults_unchanged():
    assert [p.display_name for p in set_patrol_summary_metrics()] == [
        "Patrol Count",
        "Total Distance (km)",
        "Total Duration (hrs)",
        "Patrol Days",
        "Merged Area Covered (km²)",
        "Unmerged Area Covered (km²)",
    ]


def test_leg_presets():
    active, elapsed, legs, days = set_patrol_summary_metrics(
        [
            {"metric": "active_duration", "unit": "d"},
            {"metric": "elapsed_duration"},
            {"metric": "leg_count"},
            {"metric": "member_days"},
        ]
    )
    assert (active.display_name, active.column, active.aggregator, active.new_unit) == (
        "Active Duration (days)",
        "leg_active_seconds",
        "sum",
        Unit.DAY,
    )
    assert (elapsed.display_name, elapsed.column) == ("Elapsed Duration (hrs)", "patrol_elapsed_seconds")
    assert (legs.display_name, legs.column, legs.aggregator) == ("Leg Count", "patrol_segment_id", "nunique")
    assert (days.display_name, days.column, days.aggregator, days.decimal_places) == (
        "Member-days",
        "leg_member_days",
        "sum",
        0,
    )


def test_select_track_and_leg_metrics_partition():
    params = set_patrol_summary_metrics(
        [
            {"metric": "patrol_count"},
            {"metric": "active_duration"},
            {"metric": "total_distance"},
            {"metric": "leg_count"},
        ]
    )
    assert [p.display_name for p in select_track_metrics(params)] == ["Patrol Count", "Total Distance (km)"]
    assert [p.display_name for p in select_leg_metrics(params)] == ["Active Duration (hrs)", "Leg Count"]


def test_join_summary_tables_outer_on_group_key():
    left = pd.DataFrame({"Patrol Count": [2, 1]}, index=pd.Index(["A", "B"], name="patrol_type"))
    right = pd.DataFrame({"Leg Count": [3]}, index=pd.Index(["A"], name="patrol_type"))
    joined = join_summary_tables([left, right])
    assert list(joined.columns) == ["Patrol Count", "Leg Count"]
    assert joined.loc["B", "Leg Count"] != joined.loc["B", "Leg Count"]  # NaN


def test_join_summary_tables_with_empty_right_keeps_left():
    left = pd.DataFrame({"Patrol Count": [2]}, index=pd.Index(["A"], name="patrol_type"))
    assert join_summary_tables([left, pd.DataFrame()]).equals(left)
    assert join_summary_tables([left]).equals(left)
    assert join_summary_tables([]).empty


def test_leg_metrics_end_to_end_on_leg_frame():
    legs = pd.DataFrame(
        {
            "patrol_type": ["foot", "foot", "car"],
            "patrol_segment_id": ["s1", "s3", "s9"],
            "leg_active_seconds": [4 * 3600.0, 3 * 3600.0, 3600.0],
            "patrol_elapsed_seconds": [8 * 3600.0, float("nan"), 3600.0],
            "leg_member_days": [3.0, 3.0, 1.0],
        }
    )
    params = set_patrol_summary_metrics(
        [
            {"metric": "active_duration"},
            {"metric": "elapsed_duration"},
            {"metric": "leg_count"},
            {"metric": "member_days"},
        ]
    )
    out = summarize_df(legs, select_leg_metrics(params), groupby_cols=["patrol_type"])
    assert out.loc["foot"].tolist() == [7.0, 8.0, 2, 6.0]


def test_member_days_is_nan_when_no_member_data():
    legs = pd.DataFrame(
        {
            "patrol_type": ["foot", "foot"],
            "patrol_segment_id": ["s1", "s2"],
            "leg_active_seconds": [4 * 3600.0, 3 * 3600.0],
            "leg_member_days": [float("nan"), float("nan")],
        }
    )
    params = set_patrol_summary_metrics([{"metric": "active_duration"}, {"metric": "member_days"}])
    out = summarize_df(legs, select_leg_metrics(params))
    assert pd.isna(out["Member-days"].iloc[0])
    assert out.iloc[0, 0] == 7.0


def test_join_summary_tables_merges_reset_frames_on_shared_columns():
    left = pd.DataFrame({"patrol_type": ["A", "B"], "Patrol Count": [2, 1]})
    right = pd.DataFrame({"patrol_type": ["A", "C"], "Leg Count": [3, 1]})
    joined = join_summary_tables([left, right])
    assert list(joined.columns) == ["patrol_type", "Patrol Count", "Leg Count"]
    assert sorted(joined["patrol_type"]) == ["A", "B", "C"]
    row_a = joined[joined["patrol_type"] == "A"].iloc[0]
    assert row_a["Patrol Count"] == 2 and row_a["Leg Count"] == 3


def test_join_summary_tables_ungrouped_one_row_frames_join_on_index():
    joined = join_summary_tables([pd.DataFrame({"Patrol Count": [2]}), pd.DataFrame({"Leg Count": [3]})])
    assert joined.iloc[0].tolist() == [2, 3]


def test_join_summary_tables_ignores_key_only_table():
    # No leg metric selected: the leg side carries only the group keys, one of which (C) has
    # no track row. It must not add an all-blank row.
    left = pd.DataFrame({"patrol_type": ["A", "B"], "Patrol Count": [2, 1]})
    key_only = pd.DataFrame({"patrol_type": ["A", "C"]})
    assert join_summary_tables([left, key_only]).equals(left)
    assert join_summary_tables([key_only, left]).equals(left)
    # index mode: a key-only summary has no columns at all
    left_idx = pd.DataFrame({"Patrol Count": [2]}, index=pd.Index(["A"], name="patrol_type"))
    key_only_idx = pd.DataFrame(index=pd.Index(["A", "C"], name="patrol_type"))
    assert join_summary_tables([left_idx, key_only_idx]).equals(left_idx)
    # every table key-only: the first survives, so an all-key run still renders its rows
    assert join_summary_tables([key_only, pd.DataFrame({"patrol_type": ["D"]})]).equals(key_only)
