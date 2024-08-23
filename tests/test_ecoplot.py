import pytest
import numpy as np
import pandas as pd
from ecoscope.plotting.plot import EcoPlotData, ecoplot, mcp, nsd, speed, stacked_bar_chart, pie_chart


@pytest.fixture
def chart_df():
    df = pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "category": ["A", "B", "B", "B"],
            "type": ["A", "B", "C", "D"],
            "value": [25, 40, 65, 150],
            "time": ["2024-07-22", "2024-07-22", "2024-07-22", "2024-07-21"],
        }
    )
    df.set_index("id", inplace=True)
    return df


def test_ecoplot(movebank_relocations):
    traj = movebank_relocations.trajectories.from_relocations(movebank_relocations)
    epd = EcoPlotData(traj.groupby("groupby_col"), "segment_start", "speed_kmhr", line=dict(color="blue"))
    figure = ecoplot([epd], "EcoPlot")

    habiba = traj.loc[traj["groupby_col"] == "Habiba"]
    salif = traj.loc[traj["groupby_col"] == "Salif Keita"]

    assert len(figure.data) == 2

    assert figure.data[0].name == "Habiba"
    assert np.equal(figure.data[0].x, habiba["segment_start"].array).all()
    assert np.equal(figure.data[0].y, habiba["speed_kmhr"].array).all()

    assert figure.data[1].name == "Salif Keita"
    assert np.equal(figure.data[1].x, salif["segment_start"].array).all()
    assert np.equal(figure.data[1].y, salif["speed_kmhr"].array).all()


def test_mcp(movebank_relocations):
    figure = mcp(movebank_relocations)

    assert len(figure.data) == 1
    assert movebank_relocations["fixtime"].iat[0] == figure.data[0].x[0]
    assert movebank_relocations["fixtime"].iat[-1] == figure.data[0].x[-1]


def test_nsd(movebank_relocations):
    figure = nsd(movebank_relocations)

    assert len(figure.data) == 1
    assert len(figure.data[0].x) == len(movebank_relocations)
    assert len(figure.data[0].y) == len(movebank_relocations)


def test_speed(movebank_relocations):
    traj = movebank_relocations.trajectories.from_relocations()
    figure = speed(traj)

    assert len(figure.data) == 1
    len(figure.data[0].x) == len(traj) * 4
    len(figure.data[0].y) == len(traj) * 4


def test_stacked_no_style(chart_df):
    gb = chart_df.groupby(["time", "category"])
    epd = EcoPlotData(gb, "time", "category", groupby_style=None)
    chart = stacked_bar_chart(epd, agg_function="count", stack_column="category")

    # we should have 2 categorical buckets
    assert len(chart.data) == 2
    assert chart.data[0].name == "A"
    assert chart.data[1].name == "B"
    # Should be the count of A and B for our 2 dates
    assert chart.data[0].y == (0, 1)
    assert chart.data[1].y == (1, 2)


def test_stacked_bar_chart_categorical(chart_df):
    groupby_style = {"A": {"marker_color": "red"}, "B": {"marker_color": "blue"}}
    style = {"marker_line_color": "black", "xperiodalignment": "middle"}
    layout_kwargs = {"plot_bgcolor": "gray", "xaxis_dtick": 86400000}

    gb = chart_df.groupby(["time", "category"])
    epd = EcoPlotData(gb, "time", "category", groupby_style=groupby_style, **style)
    chart = stacked_bar_chart(epd, agg_function="count", stack_column="category", layout_kwargs=layout_kwargs)

    # we should have 2 categorical buckets
    assert len(chart.data) == 2
    assert chart.data[0].name == "A"
    assert chart.data[1].name == "B"
    # Should be the count of A and B for our 2 dates
    assert chart.data[0].y == (0, 1)
    assert chart.data[1].y == (1, 2)
    # validate style kwargs
    assert chart.layout.plot_bgcolor == "gray"
    assert chart.data[0].xperiodalignment == chart.data[1].xperiodalignment == "middle"
    assert chart.data[0].marker.line.color == chart.data[1].marker.line.color == "black"
    assert chart.data[0].marker.color == "red"
    assert chart.data[1].marker.color == "blue"


def test_stacked_bar_chart_numerical(chart_df):
    groupby_style = {"A": {"marker_color": "yellow"}, "B": {"marker_color": "green"}}
    style = {"marker_line_color": "black", "xperiodalignment": "middle"}
    layout_kwargs = {"plot_bgcolor": "gray", "xaxis_dtick": 86400000}

    gb = chart_df.groupby(["time", "category"])
    epd = EcoPlotData(gb, "time", "value", groupby_style=groupby_style, **style)
    chart = stacked_bar_chart(epd, agg_function="sum", stack_column="category", layout_kwargs=layout_kwargs)

    # we should have 2 categorical buckets
    assert len(chart.data) == 2
    assert chart.data[0].name == "A"
    assert chart.data[1].name == "B"
    # Should be the the sum of values by A and B over time
    assert chart.data[0].y == (0, 25)
    assert chart.data[1].y == (150, 105)
    # validate style kwargs
    assert chart.layout.plot_bgcolor == "gray"
    assert chart.data[0].xperiodalignment == chart.data[1].xperiodalignment == "middle"
    assert chart.data[0].marker.line.color == chart.data[1].marker.line.color == "black"
    assert chart.data[1].marker.color == "green"


def test_pie_chart_no_style(chart_df):
    chart = pie_chart(chart_df, value_column="category")

    assert set(chart.data[0].labels) == set(["A", "B"])
    assert set(chart.data[0].values) == set([1, 3])


def test_pie_chart_categorical(chart_df):
    layout = {"piecolorway": ["red", "green", "blue"]}
    style = {"marker_line_color": "#000000", "marker_line_width": 2, "textinfo": "value"}
    chart = pie_chart(chart_df, value_column="category", style_kwargs=style, layout_kwargs=layout)

    assert chart.layout["piecolorway"] == ("red", "green", "blue")
    assert set(chart.data[0].labels) == set(["A", "B"])
    assert set(chart.data[0].values) == set([1, 3])
    assert chart.data[0].marker.line.color == "#000000"
    assert chart.data[0].marker.line.width == 2


def test_pie_chart_numerical(chart_df):
    layout = {"piecolorway": ["red", "green", "blue"]}
    style = {"marker_line_color": "#000000", "marker_line_width": 2, "textinfo": "value"}
    chart = pie_chart(chart_df, value_column="value", label_column="type", style_kwargs=style, layout_kwargs=layout)

    assert chart.layout["piecolorway"] == ("red", "green", "blue")
    assert set(chart.data[0].labels) == set(chart_df.type)
    assert set(chart.data[0].values) == set(chart_df.value)
    assert chart.data[0].marker.line.color == "#000000"
    assert chart.data[0].marker.line.width == 2
