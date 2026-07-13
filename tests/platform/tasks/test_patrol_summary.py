from ecoscope.platform.tasks.analysis._patrol_summary import (
    AreaCoveredMetric,
    CustomMetric,
    PatrolCountMetric,
    TotalDistanceMetric,
    TotalDurationMetric,
    set_patrol_summary_metrics,
)
from ecoscope.platform.tasks.analysis._summary import NumericSummaryParam
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
        [TotalDistanceMetric(metric="total_distance", unit="m"), TotalDurationMetric(metric="total_duration", unit="d")]
    )
    assert dist.display_name == "Total Distance (m)"
    assert dist.new_unit == Unit.METER
    assert dur.display_name == "Total Duration (days)"
    assert dur.new_unit == Unit.DAY


def test_set_patrol_summary_metrics_custom_passthrough():
    custom = NumericSummaryParam(display_name="Max Speed", aggregator="max", column="speed_kmhr", decimal_places=2)
    params = set_patrol_summary_metrics(
        [PatrolCountMetric(metric="patrol_count"), CustomMetric(metric="custom", param=custom)]
    )
    assert params[1] is custom


def test_set_patrol_summary_metrics_accepts_dicts():
    params = set_patrol_summary_metrics([{"metric": "area_covered", "merged": False, "swath_width_meters": 250.0}])
    assert params[0].display_name == "Unmerged Area Covered (km²)"
    assert params[0].swath_width_meters == 250.0


def test_area_covered_metric_defaults():
    m = AreaCoveredMetric(metric="area_covered")
    assert m.merged is True
    assert m.swath_width_meters == 500.0
