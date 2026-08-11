from typing import Annotated, Literal, Sequence, TypeAlias, Union

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, model_validator
from pydantic.json_schema import SkipJsonSchema
from wt_registry import register

from ecoscope.platform.indexes import (  # type: ignore[import-untyped]
    SpatialGrouper,
    UserDefinedGroupers,
)
from ecoscope.platform.tasks.analysis._patrol_summary import EncounterRateMetric
from ecoscope.platform.tasks.analysis._summary import SummaryParam
from ecoscope.platform.tasks.results._ecoplot import (
    AxisStyle,
    BarLayoutStyle,
    LineStyle,
    PlotStyle,
)


class MetricsOnlyMode(BaseModel):
    """One colored series per selected metric."""

    model_config = ConfigDict(json_schema_extra={"title": "Metrics Only"})
    mode: Annotated[Literal["metrics"], Field(default="metrics", exclude=True)] = "metrics"


class CategoryBreakdownMode(BaseModel):
    """A single metric split into one colored series per category value."""

    model_config = ConfigDict(json_schema_extra={"title": "Category"})
    mode: Annotated[Literal["category"], Field(default="category", exclude=True)] = "category"
    # Deliberately a plain string with no default: the category options (and
    # the form default) are workflow concerns, constrained per-spec via
    # rjsf-overrides — like the groupers' index_name options.
    category: Annotated[str, Field(title="Category")]


class SpatialBreakdownMode(BaseModel):
    """A single metric split into one colored series per spatial feature group region."""

    model_config = ConfigDict(json_schema_extra={"title": "Spatial Group"})
    mode: Annotated[Literal["spatial"], Field(default="spatial", exclude=True)] = "spatial"
    spatial_feature_group: Annotated[
        str,
        Field(default="", title="Spatial Feature Group"),
    ] = ""


class TimeBreakdownMode(BaseModel):
    """A single metric split into one colored series per time period, overlaid
    on a shared within-period axis for period-over-period comparison."""

    model_config = ConfigDict(json_schema_extra={"title": "Time"})
    mode: Annotated[Literal["time"], Field(default="time", exclude=True)] = "time"
    unit: Annotated[Literal["year", "month", "week", "day"], Field(default="year", title="Period")] = "year"


TrendChartMode: TypeAlias = MetricsOnlyMode | CategoryBreakdownMode | SpatialBreakdownMode | TimeBreakdownMode

_BREAKDOWN_MODES = ("category", "spatial", "time")

_FINER_INTERVAL_HINT = (
    "The Time Interval above must be smaller than the period compared here — e.g. a Month interval to compare Years."
)

# Shown by the form (ajv-errors errorMessage keyword) in place of the raw
# "must NOT have more than 1 items" when the maxItems guard below trips.
_SINGLE_METRIC_ERROR = (
    "When comparing values by category, spatial group or time, only one metric can be "
    "selected — remove the extra metric rows or set Compares to Metrics Only."
)


def _flat_conditional_mode_schema(schema: dict) -> None:
    """Flat allOf/if/then form schema for the Compares union.

    RJSF discards branch formData when an anyOf selection changes, so a typed
    spatial feature group (or category pick) would be lost on mode toggle.
    Rendering the union as a flat object with conditionally revealed fields
    keeps the values — see make_aggregation_json_schema_extra for the full
    rationale. Retained fields from a non-selected branch are ignored by
    validation.
    """
    schema.pop("anyOf", None)
    schema.update(
        {
            "type": "object",
            "title": "",
            "properties": {
                "mode": {
                    "type": "string",
                    "title": "Compares",
                    "default": "metrics",
                    "description": (
                        "What each series represents: one series per metric (Metrics Only), or the single "
                        "metric split into one series per category value, spatial region, or time period."
                    ),
                    "oneOf": [
                        {"const": "metrics", "title": "Metrics Only"},
                        {"const": "category", "title": "Category"},
                        {"const": "spatial", "title": "Spatial Group"},
                        {"const": "time", "title": "Time"},
                    ],
                },
            },
            "allOf": [
                {
                    "if": {"properties": {"mode": {"const": "category"}}},
                    "then": {
                        "properties": {
                            # Options and the form default are supplied per-spec
                            # via rjsf-overrides (the column names are workflow
                            # data-model concerns, not platform ones).
                            "category": {
                                "type": "string",
                                "title": "Category",
                                "description": "The category to split the metric by.",
                            },
                        },
                    },
                },
                {
                    "if": {"properties": {"mode": {"const": "spatial"}}},
                    "then": {
                        "properties": {
                            "spatial_feature_group": {
                                "type": "string",
                                "title": "Spatial Feature Group",
                                "description": (
                                    "The EarthRanger spatial feature group whose regions split the metric into series."
                                ),
                            },
                        },
                    },
                },
                {
                    "if": {"properties": {"mode": {"const": "time"}}},
                    "then": {
                        "properties": {
                            "unit": {
                                "type": "string",
                                "title": "Period",
                                "default": "year",
                                "description": "One series per period of this size. " + _FINER_INTERVAL_HINT,
                                "oneOf": [
                                    {"const": "year", "title": "Years"},
                                    {"const": "month", "title": "Months"},
                                    {"const": "week", "title": "Weeks"},
                                    {"const": "day", "title": "Days"},
                                ],
                            },
                        },
                    },
                },
            ],
        }
    )


def _time_interval_schema(schema: dict) -> None:
    schema.pop("enum", None)
    schema["oneOf"] = [
        {"const": "year", "title": "Year"},
        {"const": "month", "title": "Month"},
        {"const": "week", "title": "Week"},
        {"const": "day", "title": "Day"},
        {"const": "hour", "title": "Hour"},
    ]


def _config_task_json_schema(schema: dict) -> None:
    # Task-level form guard: breakdown modes cap the metric rows at one, so
    # the form blocks submission instead of the run failing at validation
    # time. The TrendChartConfig model_validator is the equivalent runtime
    # gate. NOTE: becomes @register(json_schema_extra=_config_task_json_schema)
    # once the wt-registry release (> 0.2.2) ships task-level json_schema_extra
    # (infra commit d698b2b); until then workflow specs restate this allOf at
    # the task path via rjsf-overrides.
    schema["allOf"] = [
        {
            "if": {
                "properties": {
                    "mode": {
                        "properties": {"mode": {"enum": list(_BREAKDOWN_MODES)}},
                        "required": ["mode"],
                    }
                },
                "required": ["mode"],
            },
            "then": {
                "properties": {
                    "metrics": {
                        "maxItems": 1,
                        "errorMessage": {"maxItems": _SINGLE_METRIC_ERROR},
                    }
                }
            },
        }
    ]


_DEFAULT_TREND_METRICS: tuple = ({"metric": "patrol_count"},)

_MetricAdapter: TypeAdapter = TypeAdapter(EncounterRateMetric)


class TrendChartConfig(BaseModel):
    """Validated carrier for the trend chart configuration (the runtime gate
    matching the form-time guard in _config_task_json_schema)."""

    model_config = ConfigDict(title="")

    metrics: Annotated[Sequence[EncounterRateMetric], Field(default=_DEFAULT_TREND_METRICS)] = _DEFAULT_TREND_METRICS
    time_interval: Annotated[Literal["year", "month", "week", "day", "hour"], Field(default="week")] = "week"
    mode: Annotated[Union[TrendChartMode, SkipJsonSchema[None]], Field(default=None)] = None

    @model_validator(mode="after")
    def _validate(self) -> "TrendChartConfig":
        from ecoscope.platform.tasks.results._trend_chart import INTERVAL_FINENESS

        if self.mode is None:
            self.mode = MetricsOnlyMode()
        if not isinstance(self.mode, MetricsOnlyMode) and len(self.metrics) != 1:
            raise ValueError(
                f"'Compares' requires exactly one metric; {len(self.metrics)} selected — "
                "choose Metrics Only or remove extra metric rows."
            )
        if isinstance(self.mode, CategoryBreakdownMode) and not self.mode.category.strip():
            raise ValueError("Compares 'Category' requires a category column name.")
        if isinstance(self.mode, SpatialBreakdownMode) and not self.mode.spatial_feature_group.strip():
            raise ValueError("Compares 'Spatial Group' requires a spatial feature group name.")
        if isinstance(self.mode, TimeBreakdownMode) and (
            INTERVAL_FINENESS[self.time_interval] <= INTERVAL_FINENESS[self.mode.unit]
        ):
            raise ValueError(
                f"The Time Interval ({self.time_interval}) must be smaller than the compared "
                f"Period ({self.mode.unit}) — e.g. a Month interval to compare Years."
            )
        return self


@register()
def set_trend_chart_config(
    metrics: Annotated[
        Sequence[EncounterRateMetric],
        Field(
            default=_DEFAULT_TREND_METRICS,
            title="Metrics",
            description=(
                "Metrics drawn as one series each. Keep exactly ONE metric row when Compares "
                "is set to Category, Spatial Group or Time."
            ),
        ),
    ] = _DEFAULT_TREND_METRICS,
    time_interval: Annotated[
        Literal["year", "month", "week", "day", "hour"],
        Field(
            default="week",
            title="Time Interval",
            description="Choose the time interval for the x-axis to control how metrics are summarized over time.",
            json_schema_extra=_time_interval_schema,
        ),
    ] = "week",
    mode: Annotated[
        Union[TrendChartMode, SkipJsonSchema[None]],
        Field(default=None, json_schema_extra=_flat_conditional_mode_schema),
    ] = None,
) -> TrendChartConfig:
    return TrendChartConfig(metrics=metrics, time_interval=time_interval, mode=mode)


@register()
def get_trend_chart_metrics(
    config: Annotated[TrendChartConfig, Field(title="")],
) -> Annotated[list[SummaryParam], Field(description="Summary metric parameters")]:
    validated = [m if isinstance(m, BaseModel) else _MetricAdapter.validate_python(m) for m in config.metrics]
    return [m.to_summary_param() for m in validated]


@register()
def get_trend_chart_time_interval(
    config: Annotated[TrendChartConfig, Field(title="")],
) -> Literal["year", "month", "week", "day", "hour"]:
    return config.time_interval


@register()
def get_chart_mode_category_column(
    config: Annotated[TrendChartConfig, Field(title="")],
) -> str | None:
    if isinstance(config.mode, CategoryBreakdownMode):
        return config.mode.category
    if isinstance(config.mode, SpatialBreakdownMode):
        return SpatialGrouper(spatial_index_name=config.mode.spatial_feature_group).index_name
    return None


@register()
def get_chart_mode_time_unit(
    config: Annotated[TrendChartConfig, Field(title="")],
) -> str | None:
    if isinstance(config.mode, TimeBreakdownMode):
        return config.mode.unit
    return None


@register()
def get_chart_mode_spatial_groupers(
    config: Annotated[TrendChartConfig, Field(title="")],
) -> UserDefinedGroupers:
    if isinstance(config.mode, SpatialBreakdownMode):
        return [SpatialGrouper(spatial_index_name=config.mode.spatial_feature_group)]
    return []


# Default palette: the named colormap the patrols workflow uses for its
# events chart (apply_color_map colormap: tab20b).
_DEFAULT_PALETTE = "tab20b"


class BarChartStyle(BaseModel):
    """Bars per time bucket, arranged per barmode."""

    model_config = ConfigDict(json_schema_extra={"title": "Bar"})
    chart_type: Annotated[Literal["bar"], Field(default="bar", exclude=True)] = "bar"
    barmode: Annotated[Literal["group", "stack"], Field(default="group", title="Bar Mode")] = "group"
    show_value_labels: Annotated[bool, Field(default=True, title="Show value labels on each bar")] = True


class LineChartStyle(BaseModel):
    """Lines per time bucket, optionally smoothed/dashed/marked."""

    model_config = ConfigDict(json_schema_extra={"title": "Line"})
    chart_type: Annotated[Literal["line"], Field(default="line", exclude=True)] = "line"
    line_shape: Annotated[Literal["linear", "spline"], Field(default="linear", title="Line Shape")] = "linear"
    line_dash: Annotated[Literal["solid", "dash", "dot"], Field(default="solid", title="Line Style")] = "solid"
    show_markers: Annotated[bool, Field(default=True, title="Mark each data point on the line")] = True


ChartTypeStyle: TypeAlias = BarChartStyle | LineChartStyle


def _flat_conditional_chart_schema(schema: dict) -> None:
    """Flat allOf/if/then form schema for the chart-type union: bar options
    (bar mode, value labels) are revealed only for bar charts and line
    options (line shape, line style, markers) only for line charts
    (aggregation pattern — fields retained from the non-selected branch are
    ignored by validation)."""
    schema.pop("anyOf", None)
    schema.update(
        {
            "type": "object",
            "title": "",
            "properties": {
                "chart_type": {
                    "type": "string",
                    "title": "Chart Type",
                    "default": "bar",
                    "description": "Render the trend as bars or lines.",
                    "oneOf": [
                        {"const": "bar", "title": "Bar"},
                        {"const": "line", "title": "Line"},
                    ],
                },
            },
            "allOf": [
                {
                    "if": {"properties": {"chart_type": {"const": "bar"}}},
                    "then": {
                        "properties": {
                            "barmode": {
                                "type": "string",
                                "title": "Bar Mode",
                                "default": "group",
                                "description": (
                                    "Grouped shows series side by side; Stacked piles them up "
                                    "(best when all series share a unit)."
                                ),
                                "oneOf": [
                                    {"const": "group", "title": "Grouped"},
                                    {"const": "stack", "title": "Stacked"},
                                ],
                            },
                            "show_value_labels": {
                                "type": "boolean",
                                "title": "Show value labels on each bar",
                                "default": True,
                            },
                        },
                    },
                },
                {
                    "if": {"properties": {"chart_type": {"const": "line"}}},
                    "then": {
                        "properties": {
                            "line_shape": {
                                "type": "string",
                                "title": "Line Shape",
                                "default": "linear",
                                "description": "Straight segments between points, or a smoothed curve.",
                                "oneOf": [
                                    {"const": "linear", "title": "Straight"},
                                    {"const": "spline", "title": "Smooth"},
                                ],
                            },
                            "line_dash": {
                                "type": "string",
                                "title": "Line Style",
                                "default": "solid",
                                "description": "Draw the lines solid, dashed or dotted.",
                                "oneOf": [
                                    {"const": "solid", "title": "Solid"},
                                    {"const": "dash", "title": "Dashed"},
                                    {"const": "dot", "title": "Dotted"},
                                ],
                            },
                            "show_markers": {
                                "type": "boolean",
                                "title": "Mark each data point on the line",
                                "default": True,
                            },
                        },
                    },
                },
            ],
        }
    )


def _palette_schema(schema: dict) -> None:
    for branch in schema.get("anyOf", []):
        if branch.get("type") == "string":
            branch["title"] = "Color Map"
        elif branch.get("type") == "array":
            branch["title"] = "Custom Colors"


class TrendChartStyle(BaseModel):
    """Validated carrier for the trend chart styling choices."""

    model_config = ConfigDict(title="")

    chart_type: Annotated[Literal["bar", "line"], Field(default="bar")] = "bar"
    barmode: Annotated[Literal["group", "stack"], Field(default="group")] = "group"
    show_value_labels: Annotated[bool, Field(default=True)] = True
    line_shape: Annotated[Literal["linear", "spline"], Field(default="linear")] = "linear"
    line_dash: Annotated[Literal["solid", "dash", "dot"], Field(default="solid")] = "solid"
    show_markers: Annotated[bool, Field(default=True)] = True
    palette: Annotated[str | Sequence[str], Field(default=_DEFAULT_PALETTE)] = _DEFAULT_PALETTE
    y_axis_title: Annotated[str, Field(default="")] = ""


_ChartTypeAdapter: TypeAdapter = TypeAdapter(ChartTypeStyle)


@register()
def set_trend_chart_style(
    chart: Annotated[
        Union[ChartTypeStyle, SkipJsonSchema[None]],
        Field(default=None, json_schema_extra=_flat_conditional_chart_schema),
    ] = None,
    palette: Annotated[
        str | Sequence[str],
        Field(
            default=_DEFAULT_PALETTE,
            title="Color Palette",
            description="Series colors: a named color map, or your own colors (e.g. #123456) cycled in series order.",
            json_schema_extra=_palette_schema,
        ),
    ] = _DEFAULT_PALETTE,
    y_axis_title: Annotated[
        str,
        Field(
            default="",
            title="Y-Axis Title",
            description="Optional label for the value axis. Leave empty for no label.",
        ),
    ] = "",
) -> TrendChartStyle:
    if chart is None:
        chart = BarChartStyle()
    elif isinstance(chart, dict):
        chart = _ChartTypeAdapter.validate_python(chart)
    bar = chart if isinstance(chart, BarChartStyle) else BarChartStyle()
    line = chart if isinstance(chart, LineChartStyle) else LineChartStyle()
    return TrendChartStyle(
        chart_type=chart.chart_type,
        barmode=bar.barmode,
        show_value_labels=bar.show_value_labels,
        line_shape=line.line_shape,
        line_dash=line.line_dash,
        show_markers=line.show_markers,
        palette=palette,
        y_axis_title=y_axis_title,
    )


@register()
def get_style_chart_type(
    style: Annotated[TrendChartStyle, Field(title="")],
) -> Literal["bar", "line"]:
    return style.chart_type


@register()
def get_style_barmode(
    style: Annotated[TrendChartStyle, Field(title="")],
) -> Literal["group", "stack"]:
    return style.barmode


@register()
def get_style_palette(
    style: Annotated[TrendChartStyle, Field(title="")],
) -> str | list[str]:
    return style.palette if isinstance(style.palette, str) else list(style.palette)


# Bar value labels: thousands-separated, trailing zeros trimmed.
_VALUE_LABEL_TEXTTEMPLATE = "%{y:,.2~f}"


@register()
def get_style_plot_style(
    style: Annotated[TrendChartStyle, Field(title="")],
) -> PlotStyle:
    if style.chart_type == "bar":
        if style.show_value_labels:
            return PlotStyle(texttemplate=_VALUE_LABEL_TEXTTEMPLATE, textposition="auto")
        return PlotStyle()
    line = LineStyle(
        shape="spline" if style.line_shape == "spline" else None,
        dash=style.line_dash if style.line_dash != "solid" else None,
    )
    return PlotStyle(
        mode="lines+markers" if style.show_markers else "lines",
        line=line if (line.shape or line.dash) else None,
    )


@register()
def get_style_layout_style(
    style: Annotated[TrendChartStyle, Field(title="")],
) -> BarLayoutStyle | None:
    if style.y_axis_title.strip():
        return BarLayoutStyle(yaxis=AxisStyle(title=style.y_axis_title.strip()))
    return None
