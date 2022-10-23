import logging
import os
import uuid

import numpy as np
import pandas as pd
import plotly.graph_objs as go
import shapely
from plotly.subplots import make_subplots
from sklearn.neighbors import KernelDensity

from ecoscope.io.utils import extract_voltage

logger = logging.getLogger(__name__)


class EcoPlotData:
    def __init__(self, grouped, x_col="x", y_col="y", groupby_style=None, **style):
        self.grouped = grouped
        self.x_col = x_col
        self.y_col = y_col
        self.groupby_style = {} if groupby_style is None else groupby_style
        self.style = style

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
    fig_height = n * subplot_height + 2 * subplot_height

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
        height=fig_height,
    )

    fig.update_layout(**{**dict(showlegend=False, autosize=False), **(layout_kwargs or {})})

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


def collar_event_timeline(relocations, collar_events):
    fig = go.FigureWidget()

    ys = [0]
    if not collar_events.empty:
        times = collar_events["time"].to_list()
        times.append(relocations["fixtime"][-1])
        xs = [[times[i]] * 3 + [times[i + 1]] for i in range(len(collar_events))]
        ys = [[0, i + 1, 0, 0] for i in range(len(collar_events))]
        colors = collar_events["colors"]

        for x, y, color in zip(xs, ys, colors):
            fig.add_trace(go.Scatter(x=x, y=y, line_color=color))
            fig.update_layout(
                annotations=[
                    go.layout.Annotation(x=row.time, y=i, text=f"{row.event_type}<br>{row.time.date()}")
                    for i, (_, row) in enumerate(collar_events.iterrows(), 1)
                ]
            )

    x = relocations.fixtime
    y = np.full(len(x), np.max(ys) / 10)
    fig.add_trace(go.Scatter(x=x, y=y, line_color="rgb(0,0,255)", mode="markers", marker_size=1))

    fig.update_layout(
        margin_l=0,
        margin_r=0,
        margin_t=0,
        margin_b=15,
        yaxis_visible=False,
        showlegend=False,
    )

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
    distances = relocations.distance(relocations.geometry[0]) ** 2

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


def plot_collar_voltage(
    relocations,
    start_time,
    extract_fn=extract_voltage,
    output_folder=None,
    layout_kwargs=None,
    hline_kwargs=None,
):
    # @TODO Complete black-box re-write
    assigned_range = (
        relocations["extra__subjectsource__assigned_range"]
        .apply(pd.Series)
        .add_prefix("extra.extra.subjectsource__assigned_range.")
    )
    relocations = relocations.merge(assigned_range, right_index=True, left_index=True)

    groups = relocations.groupby(by=["extra__subject__id", "extra__subjectsource__id"])

    for group, dataframe in groups:
        subject_name = relocations.loc[relocations["extra__subject__id"] == group[0]]["extra__subject__name"].unique()[
            0
        ]

        dataframe["extra__subjectsource__assigned_range__upper"] = pd.to_datetime(
            dataframe["extra__subjectsource__assigned_range"].str["upper"],
            errors="coerce",
        )
        subjectsource_upperbound = dataframe["extra__subjectsource__assigned_range__upper"].unique()

        is_source_active = subjectsource_upperbound >= start_time or pd.isna(subjectsource_upperbound)[0]
        if is_source_active:
            logger.info(subject_name)

            dataframe = dataframe.sort_values(by=["fixtime"])
            dataframe["voltage"] = np.array(dataframe.apply(extract_fn, axis=1), dtype=np.float64)

            time = dataframe[dataframe.fixtime >= start_time].fixtime.tolist()
            voltage = dataframe[dataframe.fixtime >= start_time].voltage.tolist()

            # Calculate the historic voltage
            hist_voltage = dataframe[dataframe.fixtime <= start_time].voltage.tolist()
            if hist_voltage:
                volt_upper, volt_lower = np.nanpercentile(hist_voltage, [97.5, 2.5])
                hist_voltage_mean = np.nanmean(hist_voltage)
            else:
                volt_upper, volt_lower = np.nan, np.nan
                hist_voltage_mean = None
            volt_diff = volt_upper - volt_lower
            volt_upper = np.full((len(time)), volt_upper, dtype=np.float32)
            volt_lower = np.full((len(time)), volt_lower, dtype=np.float32)

            if np.all(volt_diff == 0):
                # jitter = np.random.random_sample((len(volt_upper,)))
                volt_upper = volt_upper + 0.025 * max(volt_upper)
                volt_lower = volt_lower - 0.025 * max(volt_lower)

            if not any(hist_voltage or voltage):
                continue

            try:
                lower_y = min(np.nanmin(np.array(hist_voltage)), np.nanmin(np.array(voltage)))
                upper_y = max(np.nanmax(np.array(hist_voltage)), np.nanmax(np.array(voltage)))
            except ValueError:
                lower_y = min(hist_voltage or voltage)
                upper_y = max(hist_voltage or voltage)
            finally:
                lower_y = lower_y - 0.1 * lower_y
                upper_y = upper_y + 0.1 * upper_y

            if not len(voltage):
                continue

            if not layout_kwargs:
                layout = go.Layout(
                    xaxis={"title": "Time"},
                    yaxis={"title": "Collar Voltage"},
                    margin={"l": 40, "b": 40, "t": 50, "r": 50},
                    hovermode="closest",
                )
            else:
                layout = go.Layout(**layout_kwargs)

            # Add the current voltage
            trace = go.Scatter(
                x=time,
                y=voltage,
                fill=None,
                showlegend=True,
                mode="lines",
                line={
                    "width": 1,
                    "shape": "spline",
                },
                line_color="rgb(0,0,246)",
                marker={
                    "colorscale": "Viridis",
                    "color": voltage,
                    "colorbar": dict(title="Colorbar"),
                    "cmax": np.max(voltage),
                    "cmin": np.min(voltage),
                },
                name=subject_name,
            )

            # Add the historical lower HPD value
            trace_lower = go.Scatter(
                x=time,
                y=volt_lower,
                fill=None,
                line_color="rgba(255,255,255,0)",
                mode="lines",
                showlegend=False,
            )

            # Add the historical max HPD value
            trace_upper = go.Scatter(
                x=time,
                y=volt_upper,
                fill="tonexty",  # fill area between trace0 and trace1
                mode="lines",
                fillcolor="rgba(0,176,246,0.2)",
                line_color="rgba(255,255,255,0)",
                showlegend=True,
                name="Historic 2.5% - 97.5%",
            )

            fig = go.Figure(layout=layout)
            fig.add_trace(trace_lower)
            fig.add_trace(trace_upper)
            fig.add_trace(trace)
            if hist_voltage_mean:
                if not hline_kwargs:
                    fig.add_hline(
                        y=hist_voltage_mean,
                        line_dash="dot",
                        line_width=1.5,
                        line_color="Red",
                        annotation_text="Historic Mean",
                        annotation_position="right",
                    )
                else:
                    fig.add_hline(**hline_kwargs)
            fig.update_layout(yaxis=dict(range=[lower_y, upper_y]))
            if output_folder:
                fig.write_image(os.path.join(f"{output_folder}/_{group}_{str(uuid.uuid4())[:4]}.png"))
            else:
                fig.show()


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
