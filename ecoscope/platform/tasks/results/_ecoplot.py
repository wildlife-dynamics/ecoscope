from typing import Annotated, Literal

from pydantic import BaseModel, Field
from pydantic.json_schema import SkipJsonSchema
from wt_registry import register

from ecoscope.platform.annotations import (
    AdvancedField,
    DataFrame,
    JsonSerializableDataFrameModel,
)
from ecoscope.platform.tasks.analysis._summary import AggOperations


class SmoothingConfig(BaseModel):
    """
    Configuration for data smoothing.
    """

    method: Annotated[
        Literal["spline"],
        Field(description="The smoothing method to apply. Currently supports 'spline'."),
    ] = "spline"
    y_min: Annotated[
        float | SkipJsonSchema[None],
        AdvancedField(
            default=None,
            description="The minimum value to clamp smoothed values to. "
            "Useful for data like precipitation where values should not go below zero.",
        ),
    ] = None
    y_max: Annotated[
        float | SkipJsonSchema[None],
        AdvancedField(
            default=None,
            description="The maximum value to clamp smoothed values to.",
        ),
    ] = None
    resolution: Annotated[
        int,
        AdvancedField(
            default=10,
            description="The resolution multiplier for interpolation points. "
            "The number of output points will be len(x) * resolution.",
        ),
    ] = 10
    degree: Annotated[
        int,
        AdvancedField(
            default=3,
            description="The degree of the spline. "
            "1: Linear, 2: Quadratic, 3: Cubic (recommended), 4-5: Higher degree.",
        ),
    ] = 3


# This is actually 20 days to account for varying month length effects on bar alignment
MONTH_IN_MILLISECONDS = 1728000000
WEEK_IN_MILLISECONDS = 604800000
DAY_IN_MILLISECONDS = 86400000
HOUR_IN_MILLISECONDS = 3600000


class LineStyle(BaseModel):
    color: Annotated[str | SkipJsonSchema[None], AdvancedField(default=None)] = None
    dash: Annotated[str | SkipJsonSchema[None], AdvancedField(default=None)] = None
    shape: Annotated[str | SkipJsonSchema[None], AdvancedField(default=None)] = None


class AxisStyle(BaseModel):
    title: Annotated[str | SkipJsonSchema[None], AdvancedField(default=None)] = None
    range: Annotated[
        list[float | SkipJsonSchema[None]] | SkipJsonSchema[None],
        AdvancedField(default=None),
    ] = None


class PlotStyle(BaseModel):
    xperiodalignment: Annotated[str | SkipJsonSchema[None], AdvancedField(default=None)] = None
    marker_colors: Annotated[list[str] | SkipJsonSchema[None], AdvancedField(default=None)] = None
    textinfo: Annotated[str | SkipJsonSchema[None], AdvancedField(default=None)] = None
    line: Annotated[LineStyle | SkipJsonSchema[None], AdvancedField(default=None)] = None
    fillcolor: Annotated[str | SkipJsonSchema[None], AdvancedField(default=None)] = None
    mode: Annotated[str | SkipJsonSchema[None], AdvancedField(default=None)] = None
    name: Annotated[str | SkipJsonSchema[None], AdvancedField(default=None)] = None
    width: Annotated[int | list[int] | SkipJsonSchema[None], AdvancedField(default=None)] = None
    xperiod: Annotated[str | SkipJsonSchema[None], AdvancedField(default=None)] = None


class PlotCategoryStyle(BaseModel):
    marker_color: Annotated[str | SkipJsonSchema[None], AdvancedField(default=None)] = None
    textposition: Annotated[str | SkipJsonSchema[None], AdvancedField(default=None)] = None
    texttemplate: Annotated[str | SkipJsonSchema[None], AdvancedField(default=None)] = None


class GroupedPlotStyle(BaseModel):
    category: Annotated[str | SkipJsonSchema[None], AdvancedField(default=None)] = None
    plot_style: Annotated[PlotCategoryStyle, AdvancedField(default=PlotCategoryStyle())] = PlotCategoryStyle()


class LayoutStyle(BaseModel):
    font_size: Annotated[float | SkipJsonSchema[None], AdvancedField(default=None)] = None
    font_color: Annotated[str | SkipJsonSchema[None], AdvancedField(default=None)] = None
    font_style: Annotated[str | SkipJsonSchema[None], AdvancedField(default=None)] = None
    plot_bgcolor: Annotated[str | SkipJsonSchema[None], AdvancedField(default=None)] = None
    showlegend: Annotated[bool | SkipJsonSchema[None], AdvancedField(default=None)] = None
    hovermode: Annotated[str | SkipJsonSchema[None], AdvancedField(default=None)] = None
    legend_title: Annotated[str | SkipJsonSchema[None], AdvancedField(default=None)] = None
    title: Annotated[str | SkipJsonSchema[None], AdvancedField(default=None)] = None
    title_x: Annotated[float | SkipJsonSchema[None], AdvancedField(ge=0.0, le=1.0, default=None)] = None
    title_y: Annotated[float | SkipJsonSchema[None], AdvancedField(ge=0.0, le=1.0, default=None)] = None
    xaxis: Annotated[AxisStyle | SkipJsonSchema[None], AdvancedField(default=None)] = None
    yaxis: Annotated[AxisStyle | SkipJsonSchema[None], AdvancedField(default=None)] = None


class BarLayoutStyle(LayoutStyle):
    bargap: Annotated[float | SkipJsonSchema[None], AdvancedField(ge=0.0, le=1.0, default=None)] = None
    bargroupgap: Annotated[float | SkipJsonSchema[None], AdvancedField(ge=0.0, le=1.0, default=None)] = None


class ExportConfig(BaseModel):
    autosizable: bool = True
    fillFrame: bool = True
    responsive: bool = True
    displaylogo: bool = False
    modeBarButtonsToRemove: list[str] = [
        "zoom2d",
        "pan2d",
        "select2d",
        "lasso2d",
        "zoomIn2d",
        "zoomOut2d",
        "autoScale2d",
        "resetScale2d",
    ]


class ExportArgs(BaseModel):
    default_height: str = "100%"
    default_width: str = "100%"
    post_script: str = """
    window.parent.postMessage(
        { type: "PlotLoaded", widgetId: "{plot_id}" },
        "*",
    );
    """
    div_id: str | None = None  # If None a guid is set automatically by plotly
    config: ExportConfig = ExportConfig()


class BarConfig(BaseModel):
    column: Annotated[str, Field(description="The dataframe column to aggregate.")]
    agg_func: Annotated[AggOperations, Field(description="The aggregate function to apply.")]
    label: Annotated[str, Field(description="The label for the bar in the chart legend.")]
    show_label: Annotated[
        bool,
        AdvancedField(
            default=False,
            description="Whether to show the value label on top of the bar.",
        ),
    ] = False
    style: Annotated[
        PlotCategoryStyle | SkipJsonSchema[None],
        AdvancedField(default=None, description="The style parameters for the category"),
    ] = None


class EcoplotConfig(BaseModel):
    x_col: Annotated[str, Field(description="The dataframe column to plot in the x axis.")]
    y_col: Annotated[str, Field(description="The dataframe column to plot in the y axis.")]
    color_col: Annotated[
        str | SkipJsonSchema[None],
        AdvancedField(
            default=None,
            description="The name of the dataframe column to color each plot group with.",
        ),
    ] = None
    plot_style: Annotated[
        PlotStyle | SkipJsonSchema[None],
        Field(description="Style arguments passed to plotly.graph_objects.Scatter."),
    ] = PlotStyle()


@register()
def draw_ecoplot(
    dataframe: DataFrame[JsonSerializableDataFrameModel],
    group_by: Annotated[str, Field(description="The dataframe column to group by.")],
    ecoplot_configs: Annotated[list[EcoplotConfig], Field(description="ecoplot configs.")],
    tickformat: Annotated[
        str,
        AdvancedField(default="%b-%Y", description="The time format for timeseries data."),
    ] = "%b-%Y",
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
    Generates an EcoPlot from the provided params

    Args:
    dataframe (pd.DataFrame): The input dataframe.
    group_by (str): The dataframe column to group by.
    ecoplot_configs (list[EcoplotConfig]): the ecoplot configs
    widget_id (str): The id of the dashboard widget that this tile layer belongs to.
        If set this MUST match the widget title as defined downstream in create_widget tasks

    Returns:
    The generated plot html as a string
    """
    import ecoscope.plotting as plotting

    grouped = dataframe.groupby(group_by)

    data = []
    for config in ecoplot_configs:
        data.append(
            plotting.EcoPlotData(
                grouped=grouped,
                x_col=config.x_col,
                y_col=config.y_col,
                color_col=config.color_col,
                **(config.plot_style.model_dump(exclude_none=True) if config.plot_style else {}),
            )
        )

    plot = plotting.ecoplot(
        data=data,
        tickformat=tickformat,
    )

    return plot.to_html(**ExportArgs(div_id=widget_id).model_dump(exclude_none=True))


@register()
def draw_time_series_bar_chart(
    dataframe: DataFrame[JsonSerializableDataFrameModel],
    x_axis: Annotated[str, Field(description="The dataframe column to plot in the x/time axis.")],
    y_axis: Annotated[str, Field(description="The dataframe column to plot in the y axis.")],
    category: Annotated[str, Field(description="The dataframe column to stack in the y axis.")],
    agg_function: Annotated[
        AggOperations,
        Field(description="The aggregate function to apply to the group."),
    ],
    time_interval: Annotated[
        Literal["year", "month", "week", "day", "hour"],
        Field(),
    ],
    color_column: Annotated[
        str | SkipJsonSchema[None],
        AdvancedField(
            default=None,
            description="The name of the dataframe column to color bars with.",
        ),
    ] = None,
    plot_style: Annotated[
        PlotStyle | SkipJsonSchema[None],
        AdvancedField(default=None, description="Additional style kwargs passed to go.Bar()."),
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
    Generates a stacked time series bar chart from the provided params

    Args:
    dataframe (pd.DataFrame): The input dataframe.
    x_axis (str): The dataframe column to plot in the x axis.
    y_axis (str): The dataframe column to plot in the y axis.
    category (str): The dataframe column to stack in the y axis.
    agg_function (str): The aggregate function to apply to the group.
    time_interval (str): Sets the time interval of the x axis.
    color_column (str): The name of the dataframe column to color bars with.
    plot_style (PlotStyle): Style arguments passed to plotly.graph_objects.Bar and applied to all groups.
    layout_style (LayoutStyle): Additional kwargs passed to plotly.go.Figure(layout).
    widget_id (str): The id of the dashboard widget that this tile layer belongs to.
        If set this MUST match the widget title as defined downstream in create_widget tasks

    Returns:
    The generated chart html as a string
    """
    import datetime

    from ecoscope.plotting import EcoPlotData, stacked_bar_chart

    layout_kws = layout_style.model_dump(exclude_none=True) if layout_style else {}
    plot_style = plot_style if plot_style else PlotStyle()

    match time_interval:
        case "year":
            dataframe["truncated_time"] = dataframe[x_axis].apply(lambda x: datetime.datetime(x.year, 1, 1))
            layout_kws["xaxis_dtick"] = "M12"
        case "month":
            dataframe["truncated_time"] = dataframe[x_axis].apply(lambda x: datetime.datetime(x.year, x.month, 1))
            layout_kws["xaxis_dtick"] = "M1"
            plot_style.width = MONTH_IN_MILLISECONDS
            plot_style.xperiod = "M1"
            plot_style.xperiodalignment = "start"
        case "week":
            dataframe["truncated_time"] = dataframe[x_axis].apply(
                lambda x: datetime.datetime(x.year, x.month, x.day) - datetime.timedelta(x.day_of_week)
            )
            layout_kws["xaxis_dtick"] = WEEK_IN_MILLISECONDS
        case "day":
            dataframe["truncated_time"] = dataframe[x_axis].apply(lambda x: datetime.datetime(x.year, x.month, x.day))
            layout_kws["xaxis_dtick"] = DAY_IN_MILLISECONDS
        case "hour":
            dataframe["truncated_time"] = dataframe[x_axis].apply(
                lambda x: datetime.datetime(x.year, x.month, x.day, x.hour)
            )
            layout_kws["xaxis_dtick"] = HOUR_IN_MILLISECONDS
        case _:
            raise NotImplementedError(f"Unsupported time_interval: {time_interval}")

    grouped = dataframe.groupby(["truncated_time", category])

    data = EcoPlotData(
        grouped=grouped,
        x_col="truncated_time",
        y_col=y_axis,
        color_col=color_column,
        **(plot_style.model_dump(exclude_none=True) if plot_style else {}),
    )

    plot = stacked_bar_chart(
        data=data,
        agg_function=agg_function,
        stack_column=category,
        layout_kwargs=layout_kws,
    )

    return plot.to_html(**ExportArgs(div_id=widget_id).model_dump(exclude_none=True))


@register()
def draw_bar_chart(
    dataframe: DataFrame[JsonSerializableDataFrameModel],
    bar_chart_configs: Annotated[
        list[BarConfig],
        Field(description="Bar chart configuration.", title="Bar Chart Configuration"),
    ],
    category: Annotated[
        str,
        Field(description="The column name in the dataframe to group by and use as the x-axis categories."),
    ],
    layout_kwargs: Annotated[
        LayoutStyle | SkipJsonSchema[None],
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
    Generates a bar chart from the provided params

    Args:
    dataframe (pd.DataFrame): The input dataframe.
    bar_configs (list[BarConfig]): a list of BarConfigs
        specifying the the bar chart labels, columns, and functions for aggregation.
    category (str): The column name in the dataframe to group by and use as the x-axis categories.
    layout_kwargs (LayoutStyle): Additional styling options passed to plotly.go.Figure(layout).

    Returns:
    The generated chart html as a string
    """
    import ecoscope.plotting as ecoplot

    bar_configs = [
        ecoplot.BarConfig(
            column=config.column,
            agg_func=config.agg_func,
            label=config.label,
            show_label=config.show_label,
            style=config.style.model_dump(exclude_none=True) if config.style else {},
        )
        for config in bar_chart_configs
    ]
    plot = ecoplot.bar_chart(
        data=dataframe,
        bar_configs=bar_configs,
        category=category,
        layout_kwargs=layout_kwargs.model_dump(exclude_none=True) if layout_kwargs else {},
    )

    return plot.to_html(**ExportArgs(div_id=widget_id).model_dump(exclude_none=True))


@register()
def draw_line_chart(
    dataframe: DataFrame[JsonSerializableDataFrameModel],
    x_column: Annotated[str, Field(description="The dataframe column to plot in the x/time axis.")],
    y_column: Annotated[str, Field(description="The dataframe column to plot in the y/time axis.")],
    category_column: Annotated[
        str | SkipJsonSchema[None],
        Field(description="The column name in the dataframe to group by and plot separate traces."),
    ] = None,
    line_kwargs: Annotated[
        LineStyle | SkipJsonSchema[None],
        AdvancedField(
            default=None,
            description="Line style settings",
        ),
    ] = None,
    layout_kwargs: Annotated[
        LayoutStyle | SkipJsonSchema[None],
        AdvancedField(
            default=None,
            description="Additional kwargs passed to plotly.go.Figure(layout).",
        ),
    ] = None,
    smoothing: Annotated[
        SmoothingConfig | SkipJsonSchema[None],
        AdvancedField(
            default=None,
            description=(
                "Configuration for line smoothing. When set, creates a smoothed line with original data point markers."
            ),
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
    Generates a line chart from the provided params

    Args:
    dataframe (pd.DataFrame): The input dataframe.
    x_column (str): The dataframe column to plot in the x/time axis.
    y_column (str): The dataframe column to plot in the y/time axis.
    category_column (str): The column name in the dataframe to group by and plot separate traces.
    line_kwargs (LineStyle): Additional styling options passed to each line of the chart.
    layout_kwargs (LayoutStyle): Additional styling styling options passed to plotly.go.Figure(layout).
    smoothing (SmoothingConfig): Configuration for line smoothing. When set, creates two layers:
        a smoothed line and original data point markers.

    Returns:
    The generated chart html as a string
    """
    import ecoscope.plotting as ecoplot
    from ecoscope.analysis.smoothing import (
        SmoothingConfig as EcoSmoothingConfig,
    )

    smoothing_config = None
    if smoothing is not None:
        smoothing_config = EcoSmoothingConfig(
            method=smoothing.method,
            y_min=smoothing.y_min,
            y_max=smoothing.y_max,
            resolution=smoothing.resolution,
            degree=smoothing.degree,
        )

    plot = ecoplot.line_chart(
        data=dataframe,
        x_column=x_column,
        y_column=y_column,
        category_column=category_column,
        line_kwargs=line_kwargs.model_dump(exclude_none=True) if line_kwargs else {},
        layout_kwargs=layout_kwargs.model_dump(exclude_none=True) if layout_kwargs else {},
        smoothing=smoothing_config,
    )

    return plot.to_html(**ExportArgs(div_id=widget_id).model_dump(exclude_none=True))


@register()
def draw_pie_chart(
    dataframe: DataFrame[JsonSerializableDataFrameModel],
    value_column: Annotated[
        str,
        Field(description="The name of the dataframe column to pull slice values from."),
    ],
    label_column: Annotated[
        str | SkipJsonSchema[None],
        AdvancedField(
            default=None,
            description=(
                "The name of the dataframe column to label slices with,"
                " required if the data in value_column is numeric."
            ),
        ),
    ] = None,
    color_column: Annotated[
        str | SkipJsonSchema[None],
        AdvancedField(
            default=None,
            description="The name of the dataframe column to color slices with.",
        ),
    ] = None,
    plot_style: Annotated[
        PlotStyle | SkipJsonSchema[None],
        AdvancedField(default=None, description="Additional style kwargs passed to go.Pie()."),
    ] = None,
    layout_style: Annotated[
        LayoutStyle | SkipJsonSchema[None],
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
    Generates a pie chart from the provided params

    Args:
    dataframe (pd.DataFrame): The input dataframe.
    value_column (str): The name of the dataframe column to pull slice values from.
    label_column (str): The name of the dataframe column to label slices with,
        required if the data in value_column is numeric.
    plot_style (PlotStyle): Additional style kwargs passed to go.Pie().
    layout_style (LayoutStyle): Additional kwargs passed to plotly.go.Figure(layout).
    widget_id (str): The id of the dashboard widget that this tile layer belongs to.
        If set this MUST match the widget title as defined downstream in create_widget tasks

    Returns:
    The generated chart html as a string
    """
    from ecoscope.plotting import pie_chart

    plot = pie_chart(
        data=dataframe,
        value_column=value_column,
        label_column=label_column,
        color_column=color_column,
        style_kwargs=plot_style.model_dump(exclude_none=True) if plot_style else {},
        layout_kwargs=layout_style.model_dump(exclude_none=True) if layout_style else {},
    )

    return plot.to_html(**ExportArgs(div_id=widget_id).model_dump(exclude_none=True))


@register()
def draw_historic_timeseries(
    dataframe: DataFrame[JsonSerializableDataFrameModel],
    current_value_column: Annotated[
        str,
        Field(description="The name of the dataframe column to pull slice values from"),
    ],
    current_value_title: Annotated[
        str,
        Field(description="The title shown in the plot legend for current value"),
    ],
    historic_min_column: Annotated[
        str | SkipJsonSchema[None],
        Field(description="The name of the dataframe column to pull historic min values from"),
    ] = None,
    historic_max_column: Annotated[
        str | SkipJsonSchema[None],
        Field(description="The name of the dataframe column to pull historic max values from"),
    ] = None,
    historic_band_title: Annotated[
        str | SkipJsonSchema[None],
        Field(description="The title shown in the plot legend for historic band"),
    ] = "Historic Min-Max",
    historic_mean_column: Annotated[
        str | SkipJsonSchema[None],
        Field(description="The name of the dataframe column to pull historic mean values from"),
    ] = None,
    historic_mean_title: Annotated[
        str | SkipJsonSchema[None],
        Field(description="The title shown in the plot legend for historic mean values"),
    ] = "Historic Mean",
    layout_style: Annotated[
        LayoutStyle | SkipJsonSchema[None],
        AdvancedField(
            default=None,
            description="Additional kwargs passed to plotly.go.Figure(layout).",
        ),
    ] = None,
    upper_lower_band_style: Annotated[
        PlotStyle | SkipJsonSchema[None],
        AdvancedField(
            default=None,
            description="Additional kwargs for upper_lower_band passed to plotly.graph_objects.Scatter.",
        ),
    ] = PlotStyle(mode="lines", line=LineStyle(color="green")),
    historic_mean_style: Annotated[
        PlotStyle | SkipJsonSchema[None],
        AdvancedField(
            default=None,
            description="Additional kwargs passed for historic_mean to plotly.graph_objects.Scatter.",
        ),
    ] = None,
    current_value_style: Annotated[
        PlotStyle | SkipJsonSchema[None],
        AdvancedField(
            default=None,
            description="Additional kwargs for current_value passed to plotly.graph_objects.Scatter.",
        ),
    ] = None,
    time_column: Annotated[
        str | SkipJsonSchema[None],
        Field(description="The name of the dataframe column to pull historic max values from"),
    ] = "img_date",
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
    Creates a timeseries plot compared with historical values
    Parameters
    ----------
    df: pd.Dataframe
        The data to plot
    current_value_column: str
        The name of the dataframe column to pull slice values from
    current_value_title: str
        The title of the current value
    historic_min_column: str
        The name of the dataframe column to pull historic min values from.
        historic_min_column and historic_max_column should exist together.
    historic_max_column: str
        The name of the dataframe column to pull historic max values from.
        historic_min_column and historic_max_column should exist together.
    historic_mean_column: str
        The name of the dataframe column to pull historic mean values from
    layout_kwargs: dict
        Additional kwargs passed to plotly.go.Figure(layout)
    upper_lower_band_style: PlotStyle
        Additional kwargs for upper_lower_band passed to plotly.graph_objects.Scatter
    historic_mean_style: PlotStyle
        Additional kwargs passed for historic_mean to plotly.graph_objects.Scatter
    current_value_style: PlotStyle
        Additional kwargs for current_value passed to plotly.graph_objects.Scatter
    time_column: str
        The name of the dataframe column to pull time values from
    widget_id str: The id of the dashboard widget that this tile layer belongs to.
        If set this MUST match the widget title as defined downstream in create_widget tasks
    Returns
    -------
    fig : The generated chart html as a string
    """
    from ecoscope.plotting.plot import draw_historic_timeseries

    if historic_mean_style is None:
        historic_mean_style = PlotStyle(mode="lines", line=LineStyle(color="green", dash="dot"))
    if current_value_style is None:
        current_value_style = PlotStyle(mode="lines", line=LineStyle(color="navy"))

    fig = draw_historic_timeseries(
        dataframe,
        current_value_column=current_value_column,
        current_value_title=current_value_title,
        time_column=time_column,  # type: ignore[arg-type]
        historic_min_column=historic_min_column,
        historic_max_column=historic_max_column,
        historic_band_title=historic_band_title,  # type: ignore[arg-type]
        historic_mean_column=historic_mean_column,
        historic_mean_title=historic_mean_title,  # type: ignore[arg-type]
        layout_kwargs=layout_style.model_dump(exclude_none=True) if layout_style else {},
        upper_lower_band_style=upper_lower_band_style.model_dump(exclude_none=True) if upper_lower_band_style else {},
        historic_mean_style=historic_mean_style.model_dump(exclude_none=True) if historic_mean_style else {},
        current_value_style=current_value_style.model_dump(exclude_none=True) if current_value_style else {},
    )

    return fig.to_html(**ExportArgs(div_id=widget_id).model_dump(exclude_none=True))
