from typing import Annotated, Literal

from pydantic import Field
from pydantic.json_schema import SkipJsonSchema
from wt_registry import register

from ecoscope.platform.annotations import (
    AdvancedField,
    DataFrame,
    JsonSerializableDataFrameModel,
)
from ecoscope.platform.tasks.analysis._summary import (
    SummaryParam,
    summarize_column,
)
from ecoscope.platform.tasks.results._ecoplot import (
    DAY_IN_MILLISECONDS,
    HOUR_IN_MILLISECONDS,
    MONTH_IN_MILLISECONDS,
    WEEK_IN_MILLISECONDS,
    BarLayoutStyle,
    ExportArgs,
    PlotStyle,
)

TimeInterval = Literal["year", "month", "week", "day", "hour"]
TimeBreakdownUnit = Literal["year", "month", "week", "day"]

# Coarse -> fine. A time breakdown requires time_interval strictly finer than
# the breakdown unit.
INTERVAL_FINENESS: dict[str, int] = {"year": 0, "month": 1, "week": 2, "day": 3, "hour": 4}

_INTERVAL_DTICKS = {
    "year": "M12",
    "month": "M1",
    "week": WEEK_IN_MILLISECONDS,
    "day": DAY_IN_MILLISECONDS,
    "hour": HOUR_IN_MILLISECONDS,
}

# 2000 is a leap year, so Feb 29 buckets survive the rebase.
_REBASE_YEAR = 2000

_PERIOD_LABEL_FORMATS = {
    "year": "%Y",
    "month": "%b %Y",
    "week": "Week of %d %b %Y",
    "day": "%d %b %Y",
}

# x tick/hover formats for the shared within-period axis, keyed by
# (time_breakdown, time_interval). Week intervals are absent: they plot on an
# integer week-of-period axis (see the week branch in the time_breakdown path).
_WITHIN_PERIOD_TICKFORMATS = {
    "year": {"month": "%b", "day": "%d %b", "hour": "%d %b %H:00"},
    "month": {"day": "Day %e", "hour": "%e %H:00"},
    "week": {"day": "%a", "hour": "%a %H:00"},
    "day": {"hour": "%H:00"},
}


def _truncate(value, interval: str):
    """Floor a timestamp to the start of its interval bucket (week = Monday)."""
    import datetime

    match interval:
        case "year":
            return datetime.datetime(value.year, 1, 1)
        case "month":
            return datetime.datetime(value.year, value.month, 1)
        case "week":
            return datetime.datetime(value.year, value.month, value.day) - datetime.timedelta(int(value.day_of_week))
        case "day":
            return datetime.datetime(value.year, value.month, value.day)
        case "hour":
            return datetime.datetime(value.year, value.month, value.day, value.hour)
        case _:
            raise NotImplementedError(f"Unsupported time_interval: {interval}")


def _rebase(bucket, unit: str):
    """Map a truncated bucket onto a shared within-period datetime axis by
    dropping the period component, so buckets from different periods overlay."""
    import datetime

    match unit:
        case "year":
            return bucket.replace(year=_REBASE_YEAR)
        case "month":
            return bucket.replace(year=_REBASE_YEAR, month=1)
        case "week":
            monday = datetime.datetime(_REBASE_YEAR, 1, 3)  # a Monday
            return monday + datetime.timedelta(days=bucket.weekday(), hours=bucket.hour)
        case "day":
            return bucket.replace(year=_REBASE_YEAR, month=1, day=1)
        case _:
            raise NotImplementedError(f"Unsupported time_breakdown: {unit}")


def _resolve_palette_colors(palette: str | list[str] | None, n: int) -> list[str] | None:
    """Palette as css color strings: a hex list passes through; a string is
    resolved as a named matplotlib colormap sampled over the series count
    (the same resolution apply_color_map uses)."""
    if not palette:
        return None
    if isinstance(palette, str):
        from ecoscope.analysis.classifier import resolve_categorical_cmap_colors

        return [f"rgba({r}, {g}, {b}, {a / 255})" for r, g, b, a in resolve_categorical_cmap_colors(palette, max(n, 1))]
    return list(palette)


@register()
def draw_time_series_chart(
    dataframe: DataFrame[JsonSerializableDataFrameModel],
    x_axis: Annotated[str, Field(description="The dataframe column to plot in the x/time axis.")],
    time_interval: Annotated[
        TimeInterval,
        Field(description="The time bucket for the x axis — how metrics are summarized over time."),
    ],
    summary_params: Annotated[
        list[SummaryParam],
        Field(description="The metrics to compute per time bucket, drawn as one series each."),
    ],
    category: Annotated[
        str | SkipJsonSchema[None],
        Field(
            default=None,
            description="The column or index level to break the single metric down into one series per value.",
        ),
    ] = None,
    time_breakdown: Annotated[
        TimeBreakdownUnit | SkipJsonSchema[None],
        Field(
            default=None,
            description="Break the single metric into one series per period of this size, overlaid on a "
            "shared within-period axis. Must be coarser than time_interval; mutually exclusive with category.",
        ),
    ] = None,
    chart_type: Annotated[
        Literal["bar", "line"],
        Field(default="bar", description="Render the series as bars or lines."),
    ] = "bar",
    barmode: Annotated[
        Literal["stack", "group"],
        Field(default="group", description="How to arrange bar series within each time bucket (bar charts only)."),
    ] = "group",
    palette: Annotated[
        str | list[str] | SkipJsonSchema[None],
        AdvancedField(
            default=None,
            description="Series colors: a named matplotlib colormap or a list of colors, cycled in series order.",
        ),
    ] = None,
    plot_style: Annotated[
        PlotStyle | SkipJsonSchema[None],
        AdvancedField(default=None, description="Additional style kwargs passed to each trace."),
    ] = None,
    layout_style: Annotated[
        BarLayoutStyle | SkipJsonSchema[None],
        AdvancedField(
            default=None,
            description="Additional kwargs passed to plotly.go.Figure(layout).",
        ),
    ] = None,
    widget_id: Annotated[
        str | SkipJsonSchema[None],
        Field(
            description="""\
            The id of the dashboard widget that this tile layer belongs to.
            If set this MUST match the widget title as defined downstream in create_widget tasks
            """,
            exclude=True,
        ),
    ] = None,
) -> Annotated[str, Field()]:
    """
    Generates a time series chart (bar or line) of summary metrics.

    Each summary param is computed per time bucket via summarize_column and
    drawn as one series. With a breakdown, exactly one summary param is
    allowed and the series are instead:
    - category: one series per value of a column or index level.
    - time_breakdown: one series per period (e.g. per year), overlaid on a
      shared within-period datetime axis (period component rebased away) so
      periods can be compared; time_interval must be strictly finer. Week
      intervals overlay on an integer week-of-period axis (week 1 = the
      first 7 days of the period) so week series align across periods.

    Args:
    dataframe (pd.DataFrame): The input dataframe.
    x_axis (str): The dataframe column to plot in the x axis.
    time_interval (str): The time bucket for the x axis.
    summary_params (list[SummaryParam]): The metrics to compute per time bucket.
    category (str): The column or index level to break the single metric down by.
    time_breakdown (str): The period size to break the single metric down by.
    chart_type (str): Render the series as bars or lines.
    barmode (str): How to arrange bar series within each time bucket (bar charts only).
    palette (str | list[str]): A named matplotlib colormap or a list of colors, cycled in series order.
    plot_style (PlotStyle): Style arguments applied to every trace.
    layout_style (LayoutStyle): Additional kwargs passed to plotly.go.Figure(layout).
    widget_id (str): The id of the dashboard widget that this tile layer belongs to.
        If set this MUST match the widget title as defined downstream in create_widget tasks

    Returns:
    The generated chart html as a string
    """
    import plotly.graph_objects as go  # type: ignore[import-untyped]

    category = category or None
    time_breakdown = time_breakdown or None
    if category is not None and time_breakdown is not None:
        raise ValueError("category and time_breakdown are mutually exclusive breakdowns")
    if (category is not None or time_breakdown is not None) and len(summary_params) != 1:
        raise ValueError(
            f"'Compares' requires exactly one metric; {len(summary_params)} selected — "
            "choose Metrics Only or remove extra metric rows."
        )
    if time_breakdown is not None and INTERVAL_FINENESS[time_interval] <= INTERVAL_FINENESS[time_breakdown]:
        raise ValueError(
            f"The Time Interval ({time_interval}) must be smaller than the compared "
            f"Period ({time_breakdown}) — e.g. a Month interval to compare Years."
        )

    layout_kws = layout_style.model_dump(exclude_none=True) if layout_style else {}
    plot_style = plot_style if plot_style else PlotStyle()

    dataframe["truncated_time"] = dataframe[x_axis].apply(lambda x: _truncate(x, time_interval))
    layout_kws["xaxis_dtick"] = _INTERVAL_DTICKS[time_interval]
    # plotly hides the legend on single-trace figures, but the series name is
    # the only place the metric is identified — always show it.
    layout_kws.setdefault("showlegend", True)

    if chart_type == "bar":
        trace_style = plot_style.model_dump(exclude_none=True)

        def make_trace(x, y, name, color):
            return go.Bar(x=x, y=y, name=name, marker_color=color, **trace_style)

    else:
        # bar-only styling (period widths, bar value labels) does not apply to lines
        trace_style = plot_style.model_dump(
            exclude_none=True,
            exclude={"width", "xperiod", "xperiodalignment", "textposition", "texttemplate"},
        )
        trace_style.setdefault("mode", "lines+markers")

        def make_trace(x, y, name, color):
            return go.Scatter(x=x, y=y, name=name, line_color=color, marker_color=color, **trace_style)

    if time_breakdown is not None:
        # one series per period, overlaid on the shared within-period axis
        param = summary_params[0]
        dataframe["period_start"] = dataframe[x_axis].apply(lambda x: _truncate(x, time_breakdown))
        if time_interval == "week":
            # Calendar weeks (Monday-anchored) straddle period boundaries and
            # start on different days-of-period in different periods, so their
            # rebased buckets would never align across series. Bucket weeks
            # relative to each period's start instead — week k covers days
            # [7(k-1), 7k) of the period — on an integer week-of-period axis.
            dataframe["truncated_time"] = [
                (t.replace(tzinfo=None) - p).days // 7 + 1 for t, p in zip(dataframe[x_axis], dataframe["period_start"])
            ]
            layout_kws["xaxis_dtick"] = 1
            layout_kws["xaxis_tickprefix"] = "Week "
        else:
            dataframe["truncated_time"] = dataframe["truncated_time"].apply(lambda b: _rebase(b, time_breakdown))
        series: dict = {}
        for (period, bucket), group in dataframe.groupby(["period_start", "truncated_time"]):
            x, y = series.setdefault(period, ([], []))
            x.append(bucket)
            y.append(summarize_column(group, param))
        keys = sorted(series)
        colors = _resolve_palette_colors(palette, len(keys))
        traces = [
            make_trace(
                series[period][0],
                series[period][1],
                period.strftime(_PERIOD_LABEL_FORMATS[time_breakdown]),
                colors[i % len(colors)] if colors else None,
            )
            for i, period in enumerate(keys)
        ]
        tickformat = _WITHIN_PERIOD_TICKFORMATS[time_breakdown].get(time_interval)
        if tickformat:
            layout_kws["xaxis_tickformat"] = tickformat
            layout_kws["xaxis_hoverformat"] = tickformat
    elif category is not None:
        # one series per category value, all showing the single metric.
        # groupby resolves category as a column or an index level.
        param = summary_params[0]
        series = {}
        for (bucket, value), group in dataframe.groupby(["truncated_time", category]):
            x, y = series.setdefault(value, ([], []))
            x.append(bucket)
            y.append(summarize_column(group, param))
        keys = sorted(series)
        colors = _resolve_palette_colors(palette, len(keys))
        traces = [
            make_trace(series[value][0], series[value][1], str(value), colors[i % len(colors)] if colors else None)
            for i, value in enumerate(keys)
        ]
    else:
        groups = list(dataframe.groupby("truncated_time"))
        x_values = [bucket for bucket, _ in groups]
        colors = _resolve_palette_colors(palette, len(summary_params))
        traces = [
            make_trace(
                x_values,
                [summarize_column(group, param) for _, group in groups],
                param.display_name,
                colors[i % len(colors)] if colors else None,
            )
            for i, param in enumerate(summary_params)
        ]

    if chart_type == "bar":
        layout_kws["barmode"] = barmode
        if time_interval == "month" and (barmode == "stack" or len(traces) == 1):
            # Full-month bars: months vary in length so plotly cannot auto-size
            # a uniform bar. Safe only when bars share an x slot — a grouped
            # multi-series chart with month-wide bars would overlap neighboring
            # groups, so those keep plotly's automatic group sizing.
            for trace in traces:
                trace.update(width=MONTH_IN_MILLISECONDS, xperiod="M1", xperiodalignment="start")
    plot = go.Figure(data=traces, layout=go.Layout(**layout_kws))

    return plot.to_html(**ExportArgs(div_id=widget_id).model_dump(exclude_none=True))
