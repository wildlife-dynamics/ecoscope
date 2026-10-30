from typing import Annotated, Literal

from pydantic import BaseModel, Field
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
    ExportConfig,
    PlotStyle,
)

TimeInterval = Literal["year", "month", "week", "day", "hour"]
TimeBreakdownUnit = Literal["year", "month", "week", "day"]

# Coarse -> fine; a time breakdown needs time_interval strictly finer.
INTERVAL_FINENESS: dict[str, int] = {"year": 0, "month": 1, "week": 2, "day": 3, "hour": 4}

_INTERVAL_DTICKS = {
    "year": "M12",
    "month": "M1",
    "week": WEEK_IN_MILLISECONDS,
    "day": DAY_IN_MILLISECONDS,
    "hour": HOUR_IN_MILLISECONDS,
}

# Bucket-start frequencies matching _truncate (weeks start Monday).
_INTERVAL_FREQS = {"year": "YS", "month": "MS", "week": "W-MON", "day": "D", "hour": "h"}

# Panel layout, in pixels: the gap between stacked panels and the figure margins.
_PANEL_GAP = 8
_PANEL_MARGIN_TOP = 16
_PANEL_MARGIN_BOTTOM = 56

# 2000 is a leap year, so Feb 29 buckets survive the rebase.
_REBASE_YEAR = 2000

_PERIOD_LABEL_FORMATS = {
    "year": "%Y",
    "month": "%b %Y",
    "week": "Week of %d %b %Y",
    "day": "%d %b %Y",
}

# Keyed by (time_breakdown, time_interval); week intervals are absent — they
# plot on an integer week-of-period axis instead.
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


class ChartFigure(BaseModel):
    """One drawn series: a summary metric and how to render it."""

    metric: Annotated[SummaryParam, Field(description="The summary metric computed per x-axis bucket.")]
    chart_type: Annotated[
        Literal["bar", "line"],
        Field(default="bar", description="Render this figure as bars or a line."),
    ] = "bar"


def _bucket_range(buckets, interval: str) -> list:
    """Every bucket start from the earliest to the latest of the given buckets."""
    import pandas as pd

    return list(pd.date_range(min(buckets), max(buckets), freq=_INTERVAL_FREQS[interval]).to_pydatetime())


def _fill_buckets(x: list, y: list, full: list) -> tuple[list, list]:
    """A series reindexed onto every bucket in full, with 0 where it had none."""
    values = dict(zip(x, y))
    return full, [values.get(bucket, 0) for bucket in full]


def _css_color(value) -> str | None:
    """A color cell as a css color: an (r, g, b[, a]) 0-255 tuple, as apply_color_map and
    assign_subject_colors write, or a css color string passed through."""
    if isinstance(value, str):
        return value or None
    try:
        channels = [float(channel) for channel in value]
    except TypeError:
        return None
    if len(channels) == 3:
        return "rgb({:g}, {:g}, {:g})".format(*channels)
    if len(channels) == 4:
        return "rgba({:g}, {:g}, {:g}, {:g})".format(*channels[:3], channels[3] / 255)
    return None


def _category_colors(dataframe, category: str, color_column: str) -> dict:
    """Each category value's first parseable color in color_column."""
    colors: dict = {}
    for value, group in dataframe.groupby(category):
        color = next((c for c in map(_css_color, group[color_column]) if c is not None), None)
        if color is not None:
            colors[value] = color
    return colors


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
def draw_chart(
    dataframe: DataFrame[JsonSerializableDataFrameModel],
    x_axis: Annotated[str, Field(description="The dataframe column to plot in the x/time axis.")],
    figures: Annotated[
        list[ChartFigure],
        Field(description="The figures drawn in the plot — one series each, bars and lines freely mixed."),
    ],
    time_interval: Annotated[
        TimeInterval | SkipJsonSchema[None],
        Field(
            default=None,
            description="The time bucket for the x axis — how metrics are summarized over time. "
            "If unset, the raw x_axis values are the buckets (categorical x axis).",
        ),
    ] = None,
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
    category_layout: Annotated[
        Literal["overlay", "panels"],
        AdvancedField(
            default="overlay",
            description="How a category breakdown is drawn: every series on one chart, or one panel per "
            "category value, stacked on a shared x axis and labelled with the value.",
        ),
    ] = "overlay",
    panel_height: Annotated[
        int,
        AdvancedField(
            default=90,
            ge=40,
            description="The height of each panel in pixels. The chart grows with the number of panels and "
            "scrolls rather than squashing them.",
        ),
    ] = 90,
    fill_empty_buckets: Annotated[
        bool,
        AdvancedField(
            default=False,
            description="Draw every time bucket from the earliest to the latest in the data, at 0 where a "
            "series has no rows, so gaps show as zero rather than being bridged. Use with count or sum "
            "metrics. Requires a time_interval; period comparisons are drawn as-is.",
        ),
    ] = False,
    color_column: Annotated[
        str | SkipJsonSchema[None],
        AdvancedField(
            default=None,
            description="A column holding each category value's color, e.g. the output of "
            "assign_subject_colors. Overrides the palette for every category value that has one.",
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
    Generates a chart of summary metrics, mixing bar and line figures freely.

    Each figure's metric is computed per x-axis bucket via summarize_column
    and drawn as one series with the figure's own chart type, sharing the x
    axis. Without a time_interval, the raw x_axis values are the buckets
    instead (a categorical x axis, e.g. one bar per subject).
    With a breakdown, exactly one figure is allowed and the series
    are instead:
    - category: one series per value of a column or index level.
    - time_breakdown: one series per period (e.g. per year), overlaid on a
      shared within-period datetime axis (period component rebased away) so
      periods can be compared; time_interval must be strictly finer. Week
      intervals overlay on an integer week-of-period axis (week 1 = the
      first 7 days of the period) so week series align across periods.
    A category breakdown can instead be drawn as one stacked panel per value
    (category_layout="panels"), each panel_height pixels tall, colored per value
    from color_column. fill_empty_buckets draws empty time buckets as 0.

    Args:
    dataframe (pd.DataFrame): The input dataframe.
    x_axis (str): The dataframe column to plot in the x axis.
    figures (list[ChartFigure]): The figures drawn in the plot — a metric and
        chart type (bar or line) each.
    time_interval (str): The time bucket for the x axis; omit to bucket by raw x_axis values.
    category (str): The column or index level to break the single figure's metric down by.
    time_breakdown (str): The period size to break the single figure's metric down by.
    barmode (str): How to arrange bar series within each time bucket (bar figures only).
    palette (str | list[str]): A named matplotlib colormap or a list of colors, cycled in series order.
    plot_style (PlotStyle): Style arguments applied to every trace.
    layout_style (LayoutStyle): Additional kwargs passed to plotly.go.Figure(layout).
    category_layout (str): Draw a category breakdown overlaid on one chart, or as one panel per value.
    panel_height (int): The height of each panel in pixels.
    fill_empty_buckets (bool): Draw empty time buckets between the first and last as 0.
    color_column (str): A column holding each category value's color, overriding the palette.
    widget_id (str): The id of the dashboard widget that this tile layer belongs to.
        If set this MUST match the widget title as defined downstream in create_widget tasks

    Returns:
    The generated chart html as a string
    """
    import plotly.graph_objects as go  # type: ignore[import-untyped]

    category = category or None
    time_interval = time_interval or None
    time_breakdown = time_breakdown or None
    if category is not None and time_breakdown is not None:
        raise ValueError("category and time_breakdown are mutually exclusive breakdowns")
    if (category is not None or time_breakdown is not None) and len(figures) != 1:
        raise ValueError(
            f"'Compares' requires exactly one metric; {len(figures)} selected — "
            "choose Metrics Only or remove extra metric rows."
        )
    if time_breakdown is not None:
        if time_interval is None:
            raise ValueError(
                f"Comparing Periods ({time_breakdown}) requires a Time Interval — "
                "set one, or remove the time breakdown."
            )
        if INTERVAL_FINENESS[time_interval] <= INTERVAL_FINENESS[time_breakdown]:
            raise ValueError(
                f"The Time Interval ({time_interval}) must be smaller than the compared "
                f"Period ({time_breakdown}) — e.g. a Month interval to compare Years."
            )
    if fill_empty_buckets and time_interval is None:
        raise ValueError("fill_empty_buckets requires a time_interval to define the buckets")

    layout_kws = layout_style.model_dump(exclude_none=True) if layout_style else {}
    plot_style = plot_style if plot_style else PlotStyle()

    # Rows without an x value (e.g. NA-filled after concat) can't be bucketed.
    dataframe = dataframe.loc[dataframe[x_axis].notna()].copy()
    if time_interval is None:
        dataframe["truncated_time"] = dataframe[x_axis]
    else:
        dataframe["truncated_time"] = dataframe[x_axis].apply(lambda x: _truncate(x, time_interval))
        layout_kws["xaxis_dtick"] = _INTERVAL_DTICKS[time_interval]
    fill = fill_empty_buckets and time_breakdown is None and not dataframe.empty

    def filled(x: list, y: list) -> tuple[list, list]:
        return _fill_buckets(x, y, full_buckets) if fill else (x, y)

    if fill:
        assert time_interval is not None  # validated above
        full_buckets = _bucket_range(dataframe["truncated_time"], time_interval)
    # plotly hides the legend on single-trace figures; the series name is the
    # only place the metric is identified.
    layout_kws.setdefault("showlegend", True)

    bar_style = plot_style.model_dump(exclude_none=True)
    # bar-only styling (period widths, bar value labels) does not apply to lines
    line_style = plot_style.model_dump(
        exclude_none=True,
        exclude={"width", "xperiod", "xperiodalignment", "textposition", "texttemplate"},
    )
    line_style.setdefault("mode", "lines+markers")

    def make_trace(chart_type, x, y, name, color):
        if chart_type == "bar":
            return go.Bar(x=x, y=y, name=name, marker_color=color, **bar_style)
        return go.Scatter(x=x, y=y, name=name, line_color=color, marker_color=color, **line_style)

    if time_breakdown is not None:
        assert time_interval is not None  # validated above: breakdown requires an interval
        figure = figures[0]
        param = figure.metric
        dataframe["period_start"] = dataframe[x_axis].apply(lambda x: _truncate(x, time_breakdown))
        if time_interval == "week":
            # Calendar weeks straddle period boundaries, so rebased buckets would
            # never align across series; week k = days [7(k-1), 7k) of the period.
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
                figure.chart_type,
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
        # groupby resolves category as a column or an index level.
        figure = figures[0]
        series = {}
        for (bucket, value), group in dataframe.groupby(["truncated_time", category]):
            x, y = series.setdefault(value, ([], []))
            x.append(bucket)
            y.append(summarize_column(group, figure.metric))
        # key=str: category values from event details can mix types (ints, strings)
        keys = sorted(series, key=str)
        colors = _resolve_palette_colors(palette, len(keys))
        value_colors = _category_colors(dataframe, category, color_column) if color_column else {}
        traces = [
            make_trace(
                figure.chart_type,
                *filled(*series[value]),
                str(value),
                value_colors.get(value, colors[i % len(colors)] if colors else None),
            )
            for i, value in enumerate(keys)
        ]
    else:
        groups = list(dataframe.groupby("truncated_time"))
        x_values = [bucket for bucket, _ in groups]
        colors = _resolve_palette_colors(palette, len(figures))
        traces = [
            make_trace(
                figure.chart_type,
                *filled(x_values, [summarize_column(group, figure.metric) for _, group in groups]),
                figure.metric.display_name,
                colors[i % len(colors)] if colors else None,
            )
            for i, figure in enumerate(figures)
        ]

    bar_traces = [trace for trace in traces if isinstance(trace, go.Bar)]
    if bar_traces:
        layout_kws["barmode"] = barmode
        if time_interval == "month" and (barmode == "stack" or len(bar_traces) == 1):
            # Month-wide bars are safe only when bars share an x slot; grouped
            # multi-series bars would overlap neighboring groups.
            for trace in bar_traces:
                trace.update(width=MONTH_IN_MILLISECONDS, xperiod="M1", xperiodalignment="start")
    if category is not None and time_breakdown is None and category_layout == "panels":
        return _draw_panels(traces, keys, layout_kws, panel_height, widget_id)
    plot = go.Figure(data=traces, layout=go.Layout(**layout_kws))

    return plot.to_html(**ExportArgs(div_id=widget_id).model_dump(exclude_none=True))


def _draw_panels(traces: list, keys: list, layout_kws: dict, panel_height: int, widget_id: str | None) -> str:
    """One panel per trace, stacked on a shared x axis, each labelled with its category value.

    The figure has a fixed pixel height, so a long list of panels scrolls in its widget rather
    than being squashed to fit.
    """
    from plotly.subplots import make_subplots  # type: ignore[import-untyped]

    rows = max(len(traces), 1)
    plot_area = rows * panel_height
    plot = make_subplots(
        rows=rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=min(_PANEL_GAP / plot_area, 1 / rows) if rows > 1 else 0.0,
    )
    for row, (trace, value) in enumerate(zip(traces, keys), 1):
        plot.add_trace(trace, row=row, col=1)
        # Labels sit on a light backing, so a series running along the top stays readable.
        plot.add_annotation(
            text=str(value),
            showarrow=False,
            xref="x domain",
            yref="y domain",
            x=0.005,
            y=0.98,
            xanchor="left",
            yanchor="top",
            bgcolor="rgba(255, 255, 255, 0.8)",
            row=row,
            col=1,
        )
    height = plot_area + _PANEL_MARGIN_TOP + _PANEL_MARGIN_BOTTOM
    # A tick per bucket crowds a long panel axis (a year of days or hours), so plotly picks the
    # ticks here instead.
    layout_kws.pop("xaxis_dtick", None)
    layout_kws["showlegend"] = False  # each panel carries its own label
    plot.update_layout(height=height, margin={"t": _PANEL_MARGIN_TOP, "b": _PANEL_MARGIN_BOTTOM}, **layout_kws)
    # Panels are short: anchor each y axis at zero so heights compare honestly, with few ticks.
    plot.update_yaxes(rangemode="tozero", nticks=3)
    # fillFrame sizes the figure to the window and hides the page's overflow, which would stop a
    # chart taller than its widget from scrolling.
    return plot.to_html(
        **ExportArgs(div_id=widget_id, default_height=f"{height}px", config=ExportConfig(fillFrame=False)).model_dump(
            exclude_none=True
        )
    )
