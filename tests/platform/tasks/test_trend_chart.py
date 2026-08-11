import pandas as pd
import pytest
from pydantic import ValidationError

from ecoscope.platform.tasks.analysis._summary import (
    RatioOperand,
    RatioSummaryParam,
    StatSummaryParam,
)
from ecoscope.platform.tasks.config.set_trend_chart import (
    CategoryBreakdownMode,
    MetricsOnlyMode,
    SpatialBreakdownMode,
    TimeBreakdownMode,
    TrendChartConfig,
    get_chart_mode_category_column,
    get_chart_mode_spatial_groupers,
    get_chart_mode_time_unit,
    get_style_barmode,
    get_style_chart_type,
    get_style_layout_style,
    get_style_palette,
    get_style_plot_style,
    get_trend_chart_metrics,
    get_trend_chart_time_interval,
    set_trend_chart_config,
    set_trend_chart_style,
)
from ecoscope.platform.tasks.results._ecoplot import MONTH_IN_MILLISECONDS, PlotStyle
from ecoscope.platform.tasks.results._trend_chart import draw_time_series_chart


@pytest.fixture
def time_series_dataframe():
    return pd.DataFrame(
        {
            "category": ["A", "B", "A", "B", "B"],
            "events": [2, 3, 5, 0, 1],
            "time": pd.to_datetime(
                ["2024-05-01", "2024-05-02", "2024-06-03", "2024-06-04", "2024-06-05"],
                utc=True,
            ),
        }
    )


@pytest.fixture
def breakdown_dataframe():
    return pd.DataFrame(
        {
            "time": pd.to_datetime(["2024-05-01", "2024-05-02", "2024-06-03", "2024-06-04"], utc=True),
            "events": [2, 3, 5, 1],
            "patrol_type": ["foot", "vehicle", "foot", "vehicle"],
        }
    )


# ---------------------------------------------------------------- chart task


@pytest.mark.parametrize("widget_id", ["THIS IS A TEST ID", None])
@pytest.mark.parametrize("time_interval", ["year", "month", "week", "day", "hour"])
def test_draw_time_series_chart_metric_mode(time_series_dataframe, widget_id, time_interval):
    summary_params = [
        StatSummaryParam(display_name="Total Events", aggregator="sum", column="events"),
        StatSummaryParam(display_name="Distinct Categories", aggregator="nunique", column="category"),
    ]

    plot = draw_time_series_chart(
        dataframe=time_series_dataframe,
        x_axis="time",
        time_interval=time_interval,
        summary_params=summary_params,
        barmode="group",
        palette=["#111111", "#222222"],
        widget_id=widget_id,
    )

    assert isinstance(plot, str)
    if widget_id:
        assert widget_id in plot
    assert '"barmode":"group"' in plot
    assert "Total Events" in plot
    assert "Distinct Categories" in plot
    assert "#111111" in plot
    assert "#222222" in plot


def test_draw_time_series_chart_skips_rows_with_missing_timestamps(time_series_dataframe):
    df = time_series_dataframe.copy()
    df.loc[len(df)] = {"category": "A", "events": 9, "time": pd.NaT}
    summary_params = [StatSummaryParam(display_name="Total Events", aggregator="sum", column="events")]

    plot = draw_time_series_chart(
        dataframe=df,
        x_axis="time",
        time_interval="month",
        summary_params=summary_params,
    )

    assert isinstance(plot, str)
    assert '"y":[5,6]' in plot  # NaT row's 9 events are excluded


def test_category_breakdown_handles_mixed_type_values(time_series_dataframe):
    df = time_series_dataframe.copy()
    df["category"] = [1, "foot", 1, "foot", "foot"]
    summary_params = [StatSummaryParam(display_name="Total Events", aggregator="sum", column="events")]

    plot = draw_time_series_chart(
        dataframe=df,
        x_axis="time",
        time_interval="month",
        summary_params=summary_params,
        category="category",
    )

    assert isinstance(plot, str)
    assert '"name":"1"' in plot
    assert '"name":"foot"' in plot


def test_draw_time_series_chart_unsupported_interval(time_series_dataframe):
    summary_params = [StatSummaryParam(display_name="Total Events", aggregator="sum", column="events")]

    with pytest.raises(NotImplementedError):
        draw_time_series_chart(
            dataframe=time_series_dataframe,
            x_axis="time",
            time_interval="Not an interval",
            summary_params=summary_params,
        )


def test_draw_time_series_chart_value_labels(time_series_dataframe):
    """plot_style texttemplate/textposition pass through to every bar trace."""
    summary_params = [StatSummaryParam(display_name="Total Events", aggregator="sum", column="events")]

    plot = draw_time_series_chart(
        dataframe=time_series_dataframe,
        x_axis="time",
        time_interval="month",
        summary_params=summary_params,
        plot_style=PlotStyle(texttemplate="%{y:,.2~f}", textposition="auto"),
    )

    assert isinstance(plot, str)
    assert '"texttemplate":"%{y:,.2~f}"' in plot
    assert '"textposition":"auto"' in plot


def test_draw_time_series_chart_month_bar_width(time_series_dataframe):
    """Full-month bar width applies only when bars share an x slot (single
    series or stacked) — month-wide grouped bars would overlap neighboring
    groups, so grouped multi-series charts keep plotly's automatic sizing."""
    one_metric = [StatSummaryParam(display_name="Total Events", aggregator="sum", column="events")]
    two_metrics = one_metric + [StatSummaryParam(display_name="Mean Events", aggregator="mean", column="events")]

    def draw(summary_params, barmode):
        return draw_time_series_chart(
            dataframe=time_series_dataframe.copy(),
            x_axis="time",
            time_interval="month",
            summary_params=summary_params,
            barmode=barmode,
        )

    single = draw(one_metric, "group")
    assert '"xperiod":"M1"' in single
    assert f'"width":{MONTH_IN_MILLISECONDS}' in single
    assert '"showlegend":true' in single
    assert '"xperiod":"M1"' in draw(two_metrics, "stack")
    grouped = draw(two_metrics, "group")
    assert '"xperiod":"M1"' not in grouped
    assert f'"width":{MONTH_IN_MILLISECONDS}' not in grouped


def test_draw_time_series_chart_defaults(time_series_dataframe):
    """No palette -> plotly default colors; barmode defaults to group."""
    summary_params = [StatSummaryParam(display_name="Distinct Categories", aggregator="nunique", column="category")]

    plot = draw_time_series_chart(
        dataframe=time_series_dataframe,
        x_axis="time",
        time_interval="month",
        summary_params=summary_params,
    )

    assert isinstance(plot, str)
    assert '"barmode":"group"' in plot


def test_draw_time_series_chart_ratio_na_on_zero_denominator(time_series_dataframe):
    # denominator sums to zero in the second month bucket
    time_series_dataframe["km"] = [10.0, 6.0, 0.0, 0.0, 0.0]
    summary_params = [
        RatioSummaryParam(
            display_name="Events per Km",
            aggregator="ratio",
            numerator=RatioOperand(aggregator="sum", column="events"),
            denominator=RatioOperand(aggregator="sum", column="km"),
        ),
    ]

    plot = draw_time_series_chart(
        dataframe=time_series_dataframe,
        x_axis="time",
        time_interval="month",
        summary_params=summary_params,
    )

    assert '"y":[0.31,null]' in plot


def test_draw_time_series_chart_line_mode(time_series_dataframe):
    summary_params = [
        StatSummaryParam(display_name="Total Events", aggregator="sum", column="events"),
        StatSummaryParam(display_name="Distinct Categories", aggregator="nunique", column="category"),
    ]

    plot = draw_time_series_chart(
        dataframe=time_series_dataframe,
        x_axis="time",
        time_interval="month",
        summary_params=summary_params,
        chart_type="line",
        palette=["#111111", "#222222"],
        plot_style=PlotStyle(texttemplate="%{y:,.2~f}", textposition="auto"),
    )

    assert isinstance(plot, str)
    assert '"type":"scatter"' in plot
    assert '"mode":"lines+markers"' in plot
    assert '"barmode":"group"' not in plot
    assert '"barmode":"stack"' not in plot
    assert "#111111" in plot
    assert '"texttemplate":"%{y:,.2~f}"' not in plot


def test_draw_time_series_chart_line_shape_dash_and_markers(time_series_dataframe):
    """plot_style.line (shape/dash) and mode merge with the palette's line_color."""
    from ecoscope.platform.tasks.results._ecoplot import LineStyle

    summary_params = [StatSummaryParam(display_name="Total Events", aggregator="sum", column="events")]

    plot = draw_time_series_chart(
        dataframe=time_series_dataframe,
        x_axis="time",
        time_interval="month",
        summary_params=summary_params,
        chart_type="line",
        palette=["#111111"],
        plot_style=PlotStyle(mode="lines", line=LineStyle(shape="spline", dash="dot")),
    )

    assert isinstance(plot, str)
    assert '"shape":"spline"' in plot
    assert '"dash":"dot"' in plot
    assert '"mode":"lines"' in plot
    assert "#111111" in plot


def test_draw_time_series_chart_named_colormap_palette(time_series_dataframe):
    summary_params = [
        StatSummaryParam(display_name="Total Events", aggregator="sum", column="events"),
        StatSummaryParam(display_name="Distinct Categories", aggregator="nunique", column="category"),
    ]

    plot = draw_time_series_chart(
        dataframe=time_series_dataframe,
        x_axis="time",
        time_interval="month",
        summary_params=summary_params,
        palette="tab10",
    )

    from ecoscope.analysis.classifier import resolve_categorical_cmap_colors

    expected = [f"rgba({r}, {g}, {b}, {a / 255})" for r, g, b, a in resolve_categorical_cmap_colors("tab10", 2)]
    for color in expected:
        assert color in plot


def test_draw_time_series_chart_breakdown_by_column(breakdown_dataframe):
    summary_params = [StatSummaryParam(display_name="Total Events", aggregator="sum", column="events")]

    plot = draw_time_series_chart(
        dataframe=breakdown_dataframe,
        x_axis="time",
        time_interval="month",
        summary_params=summary_params,
        category="patrol_type",
        barmode="group",
        palette=["#111111", "#222222"],
        widget_id="breakdown-test",
    )

    assert isinstance(plot, str)
    assert "breakdown-test" in plot
    assert '"barmode":"group"' in plot
    assert '"marker":{"color":"#111111"},"name":"foot"' in plot
    assert '"marker":{"color":"#222222"},"name":"vehicle"' in plot
    assert '"y":[2,5]' in plot
    assert '"y":[3,1]' in plot


def test_draw_time_series_chart_breakdown_by_index_level(breakdown_dataframe):
    dataframe = breakdown_dataframe.set_index("patrol_type").rename_axis("SpatialGrouper_region")
    summary_params = [StatSummaryParam(display_name="Total Events", aggregator="sum", column="events")]

    plot = draw_time_series_chart(
        dataframe=dataframe,
        x_axis="time",
        time_interval="month",
        summary_params=summary_params,
        category="SpatialGrouper_region",
    )

    assert '"name":"foot"' in plot
    assert '"name":"vehicle"' in plot
    assert '"y":[2,5]' in plot
    assert '"y":[3,1]' in plot


def test_draw_time_series_chart_breakdown_requires_single_metric(breakdown_dataframe):
    summary_params = [
        StatSummaryParam(display_name="Total Events", aggregator="sum", column="events"),
        StatSummaryParam(display_name="Mean Events", aggregator="mean", column="events"),
    ]

    with pytest.raises(ValueError, match="exactly one metric"):
        draw_time_series_chart(
            dataframe=breakdown_dataframe,
            x_axis="time",
            time_interval="month",
            summary_params=summary_params,
            category="patrol_type",
        )


def test_draw_time_series_chart_empty_string_category_is_plain_metric_mode(
    breakdown_dataframe,
):
    summary_params = [
        StatSummaryParam(display_name="Total Events", aggregator="sum", column="events"),
        StatSummaryParam(display_name="Mean Events", aggregator="mean", column="events"),
    ]

    plot = draw_time_series_chart(
        dataframe=breakdown_dataframe,
        x_axis="time",
        time_interval="month",
        summary_params=summary_params,
        category="",
    )

    assert '"name":"Total Events"' in plot
    assert '"name":"Mean Events"' in plot


def test_draw_time_series_chart_time_breakdown_overlay(breakdown_dataframe):
    """unit=month, interval=day: one series per month, day buckets rebased onto
    a shared axis (Jan 2000) so the periods overlay."""
    summary_params = [StatSummaryParam(display_name="Total Events", aggregator="sum", column="events")]

    plot = draw_time_series_chart(
        dataframe=breakdown_dataframe,
        x_axis="time",
        time_interval="day",
        summary_params=summary_params,
        time_breakdown="month",
        palette=["#111111", "#222222"],
    )

    assert '"name":"May 2024"' in plot
    assert '"name":"Jun 2024"' in plot
    assert '"x":["2000-01-01T00:00:00","2000-01-02T00:00:00"]' in plot
    assert '"x":["2000-01-03T00:00:00","2000-01-04T00:00:00"]' in plot
    assert '"y":[2,3]' in plot
    assert '"y":[5,1]' in plot
    assert '"tickformat":"Day %e"' in plot


def test_draw_time_series_chart_time_breakdown_week_of_period_aligns():
    """unit=month, interval=week: weeks bucket relative to each period's start
    (not calendar Mondays), so week indices align across the period series."""
    dataframe = pd.DataFrame(
        {
            # the calendar Mondays of these weeks differ between months
            "time": pd.to_datetime(["2024-05-02", "2024-05-09", "2024-06-04", "2024-06-12"], utc=True),
            "events": [2, 3, 5, 1],
        }
    )
    summary_params = [StatSummaryParam(display_name="Total Events", aggregator="sum", column="events")]

    plot = draw_time_series_chart(
        dataframe=dataframe,
        x_axis="time",
        time_interval="week",
        summary_params=summary_params,
        time_breakdown="month",
        chart_type="line",
    )

    assert '"name":"May 2024"' in plot
    assert '"name":"Jun 2024"' in plot
    assert plot.count('"x":[1,2]') == 2
    assert '"y":[2,3]' in plot
    assert '"y":[5,1]' in plot
    assert '"tickprefix":"Week "' in plot
    assert '"dtick":1' in plot


def test_draw_time_series_chart_time_breakdown_requires_finer_interval(
    breakdown_dataframe,
):
    summary_params = [StatSummaryParam(display_name="Total Events", aggregator="sum", column="events")]

    with pytest.raises(ValueError, match="must be smaller"):
        draw_time_series_chart(
            dataframe=breakdown_dataframe,
            x_axis="time",
            time_interval="month",
            summary_params=summary_params,
            time_breakdown="month",
        )


def test_draw_time_series_chart_breakdowns_mutually_exclusive(breakdown_dataframe):
    summary_params = [StatSummaryParam(display_name="Total Events", aggregator="sum", column="events")]

    with pytest.raises(ValueError, match="mutually exclusive"):
        draw_time_series_chart(
            dataframe=breakdown_dataframe,
            x_axis="time",
            time_interval="day",
            summary_params=summary_params,
            category="patrol_type",
            time_breakdown="month",
        )


# ------------------------------------------------------------- config task


def test_set_trend_chart_config_defaults():
    config = set_trend_chart_config()

    assert isinstance(config.mode, MetricsOnlyMode)
    assert config.time_interval == "week"
    params = get_trend_chart_metrics(config)
    assert len(params) == 1
    assert params[0].display_name == "Patrol Count"
    assert get_trend_chart_time_interval(config) == "week"


def test_trend_chart_config_breakdown_requires_single_metric():
    with pytest.raises(ValidationError, match="exactly one metric"):
        TrendChartConfig(
            metrics=[{"metric": "patrol_count"}, {"metric": "total_events"}],
            mode={"mode": "category", "category": "patrol_type"},
        )


def test_trend_chart_config_time_breakdown_requires_finer_interval():
    with pytest.raises(ValidationError, match="must be smaller"):
        TrendChartConfig(time_interval="month", mode={"mode": "time", "unit": "month"})

    config = TrendChartConfig(time_interval="month", mode={"mode": "time", "unit": "year"})
    assert isinstance(config.mode, TimeBreakdownMode)


def test_trend_chart_config_branch_field_validation():
    with pytest.raises(ValidationError, match="category column name"):
        TrendChartConfig(mode={"mode": "category", "category": "  "})
    with pytest.raises(ValidationError, match="spatial feature group name"):
        TrendChartConfig(mode={"mode": "spatial"})


def test_trend_chart_config_task_schema_blocks_multi_metric_breakdowns():
    """The task-level json_schema_extra adds the form guard to the inline
    params schema; the field-level callables shape the mode conditional."""
    from ecoscope.platform.tasks.config.set_trend_chart import (
        _config_task_json_schema,
        _flat_conditional_mode_schema,
        _time_interval_schema,
    )

    schema = {"properties": {"metrics": {"type": "array"}, "mode": {"type": "object"}}}
    _config_task_json_schema(schema)
    guard = schema["allOf"][0]
    assert guard["if"]["properties"]["mode"]["properties"]["mode"]["enum"] == [
        "category",
        "spatial",
        "time",
    ]
    assert guard["then"]["properties"]["metrics"]["maxItems"] == 1
    assert "only one metric" in guard["then"]["properties"]["metrics"]["errorMessage"]["maxItems"]

    mode_schema: dict = {"anyOf": [{"$ref": "#/$defs/MetricsOnlyMode"}]}
    _flat_conditional_mode_schema(mode_schema)
    assert "anyOf" not in mode_schema
    assert mode_schema["properties"]["mode"]["title"] == "Compares"
    assert [o["title"] for o in mode_schema["properties"]["mode"]["oneOf"]] == [
        "Metrics Only",
        "Category",
        "Spatial Group",
        "Time",
    ]
    assert [list(b["then"]["properties"].keys()) for b in mode_schema["allOf"]] == [
        ["category"],
        ["spatial_feature_group"],
        ["unit"],
    ]
    assert mode_schema["allOf"][2]["then"]["properties"]["unit"]["title"] == "Period"

    interval_schema: dict = {"enum": ["year", "month", "week", "day", "hour"]}
    _time_interval_schema(interval_schema)
    assert [o["title"] for o in interval_schema["oneOf"]] == [
        "Year",
        "Month",
        "Week",
        "Day",
        "Hour",
    ]


def test_chart_mode_splitters():
    metrics_only = set_trend_chart_config()
    category = TrendChartConfig(mode={"mode": "category", "category": "patrol_type"})
    spatial = TrendChartConfig(mode={"mode": "spatial", "spatial_feature_group": "Regions"})
    time_mode = TrendChartConfig(time_interval="month", mode={"mode": "time", "unit": "year"})

    assert get_chart_mode_category_column(metrics_only) is None
    assert get_chart_mode_category_column(category) == "patrol_type"
    assert get_chart_mode_category_column(spatial) == "SpatialGrouper_Regions"
    assert get_chart_mode_category_column(time_mode) is None

    assert get_chart_mode_time_unit(metrics_only) is None
    assert get_chart_mode_time_unit(time_mode) == "year"

    assert get_chart_mode_spatial_groupers(metrics_only) == []
    (grouper,) = get_chart_mode_spatial_groupers(spatial)
    assert grouper.spatial_index_name == "Regions"

    assert isinstance(category.mode, CategoryBreakdownMode)
    assert isinstance(spatial.mode, SpatialBreakdownMode)


# -------------------------------------------------------------- style task


def test_set_trend_chart_style_defaults():
    style = set_trend_chart_style()

    assert get_style_chart_type(style) == "bar"
    assert get_style_barmode(style) == "group"
    assert get_style_palette(style) == "tab20b"
    assert get_style_plot_style(style) == PlotStyle(texttemplate="%{y:,.2~f}", textposition="auto")
    assert get_style_layout_style(style) is None


def test_trend_chart_style_accepts_custom_colors():
    style = set_trend_chart_style(palette=["#111111", "#222222"])

    assert get_style_palette(style) == ["#111111", "#222222"]


def test_trend_chart_style_chart_union_selects_barmode():
    style = set_trend_chart_style(chart={"chart_type": "bar", "barmode": "stack"})
    assert get_style_chart_type(style) == "bar"
    assert get_style_barmode(style) == "stack"

    line = set_trend_chart_style(chart={"chart_type": "line"})
    assert get_style_chart_type(line) == "line"
    assert get_style_barmode(line) == "group"

    retained = set_trend_chart_style(chart={"chart_type": "line", "barmode": "stack"})
    assert get_style_chart_type(retained) == "line"
    assert get_style_barmode(retained) == "group"


def test_trend_chart_style_bar_value_labels_toggle():
    labels_off = set_trend_chart_style(chart={"chart_type": "bar", "show_value_labels": False})
    assert get_style_plot_style(labels_off) == PlotStyle()

    line = set_trend_chart_style(chart={"chart_type": "line"})
    plot_style = get_style_plot_style(line)
    assert plot_style.texttemplate is None
    assert plot_style.textposition is None


def test_trend_chart_style_line_options():
    line = set_trend_chart_style(chart={"chart_type": "line"})
    assert get_style_plot_style(line) == PlotStyle(mode="lines+markers")

    styled = set_trend_chart_style(
        chart={
            "chart_type": "line",
            "line_shape": "spline",
            "line_dash": "dot",
            "show_markers": False,
        }
    )
    plot_style = get_style_plot_style(styled)
    assert plot_style.mode == "lines"
    assert plot_style.line is not None
    assert plot_style.line.shape == "spline"
    assert plot_style.line.dash == "dot"

    retained = set_trend_chart_style(chart={"chart_type": "bar", "line_shape": "spline", "show_markers": False})
    assert retained.line_shape == "linear"
    assert retained.show_markers is True


def test_trend_chart_style_y_axis_title():
    style = set_trend_chart_style(y_axis_title="Patrol Count")
    layout_style = get_style_layout_style(style)
    assert layout_style is not None
    assert layout_style.yaxis is not None
    assert layout_style.yaxis.title == "Patrol Count"
    assert layout_style.model_dump(exclude_none=True) == {"yaxis": {"title": "Patrol Count"}}

    assert get_style_layout_style(set_trend_chart_style(y_axis_title="   ")) is None


def test_trend_chart_style_schema_reveals_barmode_for_bars_only():
    """The chart-type field callable renders the union as a flat conditional
    (Bar Mode revealed only for bars); the palette callable titles branches."""
    from ecoscope.platform.tasks.config.set_trend_chart import (
        _flat_conditional_chart_schema,
        _palette_schema,
    )

    chart_schema: dict = {"anyOf": [{"$ref": "#/$defs/BarChartStyle"}]}
    _flat_conditional_chart_schema(chart_schema)
    assert "anyOf" not in chart_schema
    assert chart_schema["properties"]["chart_type"]["title"] == "Chart Type"
    assert [o["title"] for o in chart_schema["properties"]["chart_type"]["oneOf"]] == [
        "Bar",
        "Line",
    ]
    assert "barmode" not in chart_schema["properties"]
    bar_guard, line_guard = chart_schema["allOf"]
    assert bar_guard["if"]["properties"]["chart_type"]["const"] == "bar"
    assert bar_guard["then"]["properties"]["barmode"]["title"] == "Bar Mode"
    assert [o["title"] for o in bar_guard["then"]["properties"]["barmode"]["oneOf"]] == ["Grouped", "Stacked"]
    assert bar_guard["then"]["properties"]["show_value_labels"]["default"] is True
    assert line_guard["if"]["properties"]["chart_type"]["const"] == "line"
    assert [o["title"] for o in line_guard["then"]["properties"]["line_shape"]["oneOf"]] == ["Straight", "Smooth"]
    assert [o["title"] for o in line_guard["then"]["properties"]["line_dash"]["oneOf"]] == [
        "Solid",
        "Dashed",
        "Dotted",
    ]
    assert line_guard["then"]["properties"]["show_markers"]["default"] is True

    palette: dict = {"anyOf": [{"type": "string"}, {"items": {"type": "string"}, "type": "array"}]}
    _palette_schema(palette)
    assert [b.get("title") for b in palette["anyOf"]] == ["Color Map", "Custom Colors"]
