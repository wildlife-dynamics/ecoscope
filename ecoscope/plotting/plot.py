import numpy as np
import pandas as pd
import shapely

from ecoscope.base.utils import color_tuple_to_css

try:
    from sklearn.neighbors import KernelDensity
    import plotly.graph_objs as go
    from plotly.subplots import make_subplots
except ModuleNotFoundError:
    raise ModuleNotFoundError(
        'Missing optional dependencies required by this module. \
         Please run pip install ecoscope["plotting"]'
    )


class EcoPlotData:
    def __init__(self, grouped, x_col="x", y_col="y", color_col=None, groupby_style=None, **style):
        self.grouped = grouped
        self.x_col = x_col
        self.y_col = y_col
        self.color_col = color_col
        self.groupby_style = {} if groupby_style is None else groupby_style
        self.style = style

        if color_col:
            for group, data in grouped:
                color = color_tuple_to_css(data[color_col].unique()[0])

                # The least significant 'group' are our categories
                if not self.groupby_style.get(group[-1]):
                    self.groupby_style[group[-1]] = {"marker_color": color}
                else:
                    self.groupby_style[group[-1]]["marker_color"] = color

        # Plotting Defaults
        self.style["mode"] = self.style.get("mode", "lines+markers")


def ecoplot(
    data,
    title="",
    out_path=None,
    subplot_height=100,
    subplot_width=700,
    vertical_spacing=0.001,
    annotate_name_pos=(0.01, 0.99),
    y_title_2=None,
    layout_kwargs=None,
    **make_subplots_kwargs,
):
    groups = sorted(list(set.union(*[set(datum.grouped.groups.keys()) for datum in data])))
    datum_1 = data[0]
    datum_2 = None
    for datum in data[1:]:
        if datum.y_col != datum_1.y_col:
            datum_2 = datum
            break

    n = len(groups)

    fig = make_subplots(
        **{
            **dict(
                rows=n,
                cols=1,
                shared_xaxes="all",
                vertical_spacing=vertical_spacing,
                x_title=data[0].x_col,
                y_title=data[0].y_col,
                row_heights=list(np.repeat(subplot_height, n)),
                column_widths=[subplot_width],
                specs=[[{"secondary_y": datum_2 is not None}]] * len(groups),
            ),
            **make_subplots_kwargs,
        }
    )

    for i, name in enumerate(groups, 1):
        for datum in data:
            try:
                df = datum.grouped.get_group(name)
            except KeyError:
                continue

            timeseries = go.Scatter(
                x=df[datum.x_col],
                y=df[datum.y_col],
                name=name,
                **{**datum.style, **datum.groupby_style.get(name, {})},
            )
            fig.add_trace(
                timeseries,
                row=i,
                col=1,
                secondary_y=datum.y_col is not datum_1.y_col,
            )

    fig.update_xaxes(tickformat="%b-%Y")

    fig.layout.annotations[1]["font"]["color"] = datum_1.style.get("line", {}).get("color", "black")

    if datum_2 is not None:
        fig.layout.annotations = list(fig.layout.annotations) + [
            go.layout.Annotation(
                {
                    "font": {
                        "size": 16,
                        "color": datum_2.style.get("line", {}).get("color", "black"),
                    },
                    "showarrow": False,
                    "text": y_title_2 or datum_2.y_col,
                    "textangle": -90,
                    "x": 1,
                    "xanchor": "right",
                    "xref": "paper",
                    "xshift": +40,
                    "y": 0.5,
                    "yanchor": "middle",
                    "yref": "paper",
                },
            )
        ]

    fig["layout"].update(
        title={
            "text": title,
            "xanchor": "center",
            "x": 0.5,
        },
    )

    fig.update_layout(**{**dict(showlegend=False), **(layout_kwargs or {})})

    if annotate_name_pos is not None:
        for i, name in enumerate(groups, 1):
            fig.add_annotation(
                text=name,
                showarrow=False,
                xref="x domain",
                yref="y domain",
                x=annotate_name_pos[0],
                y=annotate_name_pos[1],
                row=i,
                col=1,
            )

    if out_path is not None:
        fig.write_image(out_path, height=n * subplot_height)

    return fig


def add_seasons(fig, season_df):
    fig = make_subplots(figure=fig, specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scatter(
            x=pd.concat([season_df.start, season_df.end]).sort_values(),
            y=season_df.season.repeat(2),
            fill="tozeroy",
            fillcolor="rgba(0,0,255,0.1)",
            mode="none",
        ),
        secondary_y=True,
    )
    fig.update_yaxes(categoryorder="array", categoryarray=["dry", "wet"])
    return fig


def mcp(relocations):
    relocations = relocations.to_crs(relocations.estimate_utm_crs())

    areas = []
    times = []
    total = shapely.geometry.GeometryCollection()
    for time, obs in relocations.groupby(pd.Grouper(key="fixtime", freq="1D"), as_index=False):
        if obs.size:
            total = total.union(obs.geometry.unary_union).convex_hull
            areas.append(total.area)
            times.append(time)

    areas = np.array(areas)
    times = np.array(times)
    times[0] = relocations["fixtime"].iat[0]
    times[-1] = relocations["fixtime"].iat[-1]

    fig = go.FigureWidget()

    fig.add_trace(go.Scatter(x=times, y=areas / (1000**2)))

    fig.update_layout(
        margin_b=15,
        margin_l=50,
        margin_r=10,
        margin_t=25,
        title="MCP Asymptote",
        yaxis_title="MCP Area (km^2)",
        showlegend=False,
    )

    return fig


def nsd(relocations):
    relocations = relocations.to_crs(relocations.estimate_utm_crs())

    times = relocations["fixtime"]
    distances = relocations.distance(relocations.geometry.iat[0]) ** 2

    fig = go.FigureWidget()

    fig.add_trace(go.Scatter(x=times, y=distances / (1000**2)))

    fig.update_layout(
        margin_b=15,
        margin_l=50,
        margin_r=10,
        margin_t=25,
        title="Net Square Displacement (NSD)",
        yaxis_title="NSD (km^2)",
        showlegend=False,
    )

    return fig


def speed(trajectory):
    times = np.column_stack(
        [
            trajectory["segment_start"],
            trajectory["segment_start"],
            trajectory["segment_end"],
            trajectory["segment_end"],
        ]
    ).flatten()
    speeds = np.column_stack(
        [
            np.zeros(len(trajectory)),
            trajectory["speed_kmhr"],
            trajectory["speed_kmhr"],
            np.zeros(len(trajectory)),
        ]
    ).flatten()

    fig = go.FigureWidget()

    fig.add_trace(go.Scatter(x=times, y=speeds))

    fig.update_layout(
        margin_b=15,
        margin_l=50,
        margin_r=10,
        margin_t=25,
        title="Speed",
        yaxis_title="Speed (km/h)",
        showlegend=False,
    )

    return fig


def plot_seasonal_dist(ndvi_vals, cuts, bandwidth=0.05, output_file=None):
    x = ndvi_vals.sort_values().to_numpy().reshape(-1, 1)
    kde = KernelDensity(kernel="gaussian", bandwidth=bandwidth).fit(x)
    dens = np.exp(kde.score_samples(x))
    fig = go.Figure(
        data=go.Scatter(
            x=x.ravel(),
            y=dens,
            fill=None,
            showlegend=False,
            mode="lines",
            line={
                "width": 1,
                "shape": "spline",
            },
        )
    )

    [
        fig.add_vline(
            x=i, line_width=3, line_dash="dash", line_color="red", annotation_text=" Cut Val: {:.2f}".format(i)
        )
        for i in cuts[1:-1]
    ]
    fig.update_layout(xaxis_title="NDVI")
    if output_file:
        fig.write_image(output_file)
    return fig


def stacked_bar_chart(data: EcoPlotData, agg_function: str, stack_column: str, layout_kwargs: dict = None):
    """
    Creates a stacked bar chart from the provided EcoPlotData object
    Parameters
    ----------
    data: ecoscope.Plotting.EcoPlotData
        The data to plot, counts categorical data.y_col values for data.x_col
    agg_function: str
        The pandas.Dataframe.aggregate() function to run ie; 'count', 'sum'
    stack_column: str
        The name of the column in the data to build stacks from, should be categorical
    layout_kwargs: dict
        Additional kwargs passed to plotly.go.Figure(layout)
    Returns
    -------
    fig : plotly.graph_objects.Figure
        The plotly bar chart
    """
    # TODO cleanup EPD defaults
    data.style.pop("mode")

    fig = go.Figure(layout=layout_kwargs)

    x_axis_name = data.x_col
    y_axis_name = data.y_col

    agg = (
        data.grouped[y_axis_name]
        .agg(agg_function)
        .to_frame(agg_function)
        .unstack(fill_value=0)
        .stack(future_stack=True)
        .reset_index()
    )

    x = agg[x_axis_name].unique()
    for category in agg[stack_column].unique():
        fig.add_trace(
            go.Bar(
                x=x,
                y=list(agg[agg[stack_column] == category][agg_function]),
                name=str(category),
                **{**data.style, **(data.groupby_style.get(category) or {})},
            )
        )

    fig.update_layout(barmode="stack")
    return fig


def pie_chart(
    data: pd.DataFrame,
    value_column: str,
    label_column: str = None,
    color_column: str = None,
    style_kwargs: dict = {},
    layout_kwargs: dict = None,
):
    """
    Creates a pie chart from the provided dataframe
    Parameters
    ----------
    data: pd.Dataframe
        The data to plot
    value_column: str
        The name of the dataframe column to pull slice values from
        If the column contains non-numeric values, it is assumed to be categorical
            and the pie slices will be a count of the occurrences of the category
    label_column: str
        The name of the dataframe column to label slices with, required if the data in value_column is numeric
    style_kwargs: dict
        Additional style kwargs passed to go.Pie()
    layout_kwargs: dict
        Additional kwargs passed to plotly.go.Figure(layout)
    Returns
    -------
    fig : plotly.graph_objects.Figure
        The plotly bar chart
    """

    if pd.api.types.is_numeric_dtype(data[value_column]):
        if label_column is not None:
            labels = data[label_column]
            values = data[value_column]
        else:
            raise ValueError("numerical values require a label column to")
    else:  # assume categorical
        labels = data[value_column].unique()
        values = data[value_column].value_counts(sort=False)

    if color_column:
        style_kwargs["marker_colors"] = [color_tuple_to_css(color) for color in data[color_column].unique()]
    # breakpoint()

    fig = go.Figure(data=go.Pie(labels=labels, values=values, **style_kwargs), layout=layout_kwargs)
    return fig
