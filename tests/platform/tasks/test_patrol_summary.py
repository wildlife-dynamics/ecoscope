import pytest
from pydantic import ValidationError

from ecoscope.platform.tasks.analysis._patrol_summary import (
    CustomMetric,
    MergedAreaCoveredMetric,
    TotalDistanceMetric,
    TotalDurationMetric,
    set_patrol_summary_metrics,
)
from ecoscope.platform.tasks.analysis._summary import StatSummaryParam
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


def test_custom_metric_schema_hides_units_behind_dependency():
    schema = CustomMetric.model_json_schema()
    assert "original_unit" not in schema["properties"]
    assert "new_unit" not in schema["properties"]
    branches = schema["dependencies"]["convert_units"]["oneOf"]
    checked = branches[1]["properties"]
    assert checked["convert_units"]["const"] is True
    assert {o["const"] for o in checked["original_unit"]["oneOf"]} == {u.value for u in Unit}


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
