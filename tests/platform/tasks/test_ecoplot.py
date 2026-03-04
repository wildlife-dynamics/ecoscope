from importlib.resources import files

import pandas as pd
import pytest

from ecoscope.platform.mock_loaders import load_parquet
from ecoscope.platform.tasks.results._ecoplot import (
    BarConfig,
    EcoplotConfig,
    LayoutStyle,
    LineStyle,
    PlotCategoryStyle,
    PlotStyle,
    SmoothingConfig,
    draw_bar_chart,
    draw_ecoplot,
    draw_historic_timeseries,
    draw_line_chart,
    draw_pie_chart,
    draw_time_series_bar_chart,
)


@pytest.fixture
def sample_dataframe():
    """Fixture to provide a sample DataFrame for testing."""
    data = {
        "value": [500, 200, 300, 150, 400],
        "category": ["A", "B", "A", "B", "B"],
        "color_column": [
            (255, 0, 0, 255),
            (0, 0, 255, 255),
            (255, 0, 0, 255),
            (0, 0, 255, 255),
            (0, 0, 255, 255),
        ],
        "time": [
            pd.to_datetime("2024-06-01", utc=True),
            pd.to_datetime("2024-06-02", utc=True),
            pd.to_datetime("2024-06-03", utc=True),
            pd.to_datetime("2024-06-04", utc=True),
            pd.to_datetime("2024-06-05", utc=True),
        ],
    }

    return pd.DataFrame(data)


@pytest.fixture
def time_series_dataframe(time_interval):
    """Fixture to provide a sample DataFrame for testing."""
    data = {
        "category": ["A", "B", "A", "B", "B"],
        "color_column": [
            (255, 0, 0, 255),
            (0, 0, 255, 255),
            (255, 0, 0, 255),
            (0, 0, 255, 255),
            (0, 0, 255, 255),
        ],
    }
    match time_interval:
        case "year":
            data["time"] = [
                pd.to_datetime("2023-06-01 15:33:00", utc=True),
                pd.to_datetime("2023-06-01 15:34:00", utc=True),
                pd.to_datetime("2024-06-02 15:36:00", utc=True),
                pd.to_datetime("2024-06-02 15:37:00", utc=True),
                pd.to_datetime("2024-06-02 15:38:00", utc=True),
            ]
        case "month":
            data["time"] = [
                pd.to_datetime("2024-05-01 15:33:00", utc=True),
                pd.to_datetime("2024-05-01 15:34:00", utc=True),
                pd.to_datetime("2024-06-02 15:36:00", utc=True),
                pd.to_datetime("2024-06-02 15:37:00", utc=True),
                pd.to_datetime("2024-06-02 15:38:00", utc=True),
            ]
        case "week":
            data["time"] = [
                pd.to_datetime("2024-05-06 15:33:00", utc=True),
                pd.to_datetime("2024-05-06 15:34:00", utc=True),
                pd.to_datetime("2024-05-14 15:36:00", utc=True),
                pd.to_datetime("2024-05-14 15:37:00", utc=True),
                pd.to_datetime("2024-05-14 15:38:00", utc=True),
            ]
        case "day":
            data["time"] = [
                pd.to_datetime("2024-05-01 15:33:00", utc=True),
                pd.to_datetime("2024-05-01 15:34:00", utc=True),
                pd.to_datetime("2024-05-02 15:36:00", utc=True),
                pd.to_datetime("2024-05-02 15:37:00", utc=True),
                pd.to_datetime("2024-05-02 15:38:00", utc=True),
            ]
        case "hour":
            data["time"] = [
                pd.to_datetime("2024-05-01 15:33:00", utc=True),
                pd.to_datetime("2024-05-01 15:34:00", utc=True),
                pd.to_datetime("2024-05-01 16:36:00", utc=True),
                pd.to_datetime("2024-05-01 16:37:00", utc=True),
                pd.to_datetime("2024-05-01 16:38:00", utc=True),
            ]

    return pd.DataFrame(data)


@pytest.fixture
def pie_dataframe():
    """Fixture to provide a sample DataFrame for testing."""
    data = {
        "value": [500, 200, 300, 150, 400],
        "category": ["A", "B", "A", "B", "C"],
        "color_column": [
            (255, 0, 0, 255),
            (0, 0, 255, 255),
            (255, 0, 0, 255),
            (0, 0, 255, 255),
            (0, 255, 0, 255),
        ],
    }

    return pd.DataFrame(data)


@pytest.mark.parametrize("widget_id", ["THIS IS A TEST ID", None])
def test_draw_ecoplot(widget_id, sample_dataframe):
    ecoplot_config = EcoplotConfig(
        x_col="time",
        y_col="value",
        color_col="color_column",
        plot_style=PlotStyle(line_style=LineStyle(color="green")),
    )
    groupby = "category"

    plot = draw_ecoplot(
        sample_dataframe,
        group_by=groupby,
        ecoplot_configs=[ecoplot_config],
        widget_id=widget_id,
    )

    assert isinstance(plot, str)
    if widget_id:
        assert widget_id in plot


@pytest.mark.parametrize("widget_id", ["THIS IS A TEST ID", None])
@pytest.mark.parametrize(
    "time_series_dataframe, time_interval",
    [
        ("year", "year"),
        ("month", "month"),
        ("week", "week"),
        ("day", "day"),
        ("hour", "hour"),
        ("year", "Not an interval"),
    ],
    indirect=["time_series_dataframe"],
)
def test_draw_time_series_bar_chart(time_series_dataframe, widget_id, time_interval):
    bar_chart_kwargs = {
        "dataframe": time_series_dataframe,
        "x_axis": "time",
        "y_axis": "category",
        "category": "category",
        "agg_function": "count",
        "time_interval": time_interval,
        "color_column": "color_column",
        "plot_style": None,
        "layout_style": LayoutStyle(title="Some Bars"),
        "widget_id": widget_id,
    }

    if time_interval == "Not an interval":
        with pytest.raises(NotImplementedError):
            draw_time_series_bar_chart(**bar_chart_kwargs)
    else:
        plot = draw_time_series_bar_chart(**bar_chart_kwargs)
        assert isinstance(plot, str)
        if widget_id:
            assert widget_id in plot


@pytest.mark.parametrize("widget_id", ["THIS IS A TEST ID", None])
def test_draw_line_chart(widget_id, sample_dataframe):
    plot = draw_line_chart(
        sample_dataframe,
        x_column="time",
        y_column="value",
    )

    assert isinstance(plot, str)

    plot = draw_line_chart(
        sample_dataframe,
        x_column="time",
        y_column="value",
        category_column="category",
        widget_id=widget_id,
    )

    assert isinstance(plot, str)
    if widget_id:
        assert widget_id in plot


def test_draw_line_chart_with_smoothing(sample_dataframe):
    smoothing = SmoothingConfig(method="spline", y_min=0, resolution=5)
    plot = draw_line_chart(
        sample_dataframe,
        x_column="time",
        y_column="value",
        smoothing=smoothing,
    )

    assert isinstance(plot, str)

    # Test with category
    plot = draw_line_chart(
        sample_dataframe,
        x_column="time",
        y_column="value",
        category_column="category",
        smoothing=smoothing,
    )

    assert isinstance(plot, str)


@pytest.mark.parametrize("widget_id", ["THIS IS A TEST ID", None])
def test_draw_bar_chart(widget_id, sample_dataframe):
    configs = [
        BarConfig(column="value", agg_func="mean", label="Mean"),
        BarConfig(column="value", agg_func="sum", label="Sum"),
    ]
    plot = draw_bar_chart(
        sample_dataframe,
        bar_chart_configs=configs,
        category="category",
        widget_id=widget_id,
    )

    assert isinstance(plot, str)
    if widget_id:
        assert widget_id in plot


@pytest.mark.parametrize("widget_id", ["THIS IS A TEST ID", None])
def test_draw_bar_chart_label(widget_id, sample_dataframe):
    configs = [
        BarConfig(
            column="value",
            agg_func="mean",
            label="Mean",
            show_label=True,
            style=PlotCategoryStyle(textposition="outside", texttemplate="%{text:.2f}"),
        ),
        BarConfig(column="value", agg_func="sum", label="Sum"),
    ]
    plot = draw_bar_chart(
        sample_dataframe,
        bar_chart_configs=configs,
        category="category",
        widget_id=widget_id,
    )

    assert isinstance(plot, str)
    if widget_id:
        assert widget_id in plot


@pytest.mark.parametrize("widget_id", ["THIS IS A TEST ID", None])
def test_draw_pie_chart(widget_id, pie_dataframe):
    plot = draw_pie_chart(
        pie_dataframe,
        value_column="value",
        label_column="category",
        color_column="color_column",
        plot_style=PlotStyle(
            textinfo="value",
        ),
        layout_style=LayoutStyle(font_color="orange", font_style="italic", font_size=20),
        widget_id=widget_id,
    )

    assert isinstance(plot, str)
    if widget_id:
        assert widget_id in plot


@pytest.fixture
def historic_dataframe():
    return load_parquet(files("ecoscope.platform.tasks.io") / "calculate-ndvi-range.example-return.parquet")


@pytest.mark.parametrize("widget_id", ["THIS IS A TEST ID", None])
def test_draw_historic_timeseries(widget_id, historic_dataframe):
    plot = draw_historic_timeseries(
        historic_dataframe,
        current_value_column="NDVI",
        current_value_title="NDVI",
        historic_min_column="min",
        historic_max_column="max",
        historic_mean_column="mean",
        widget_id=widget_id,
    )

    assert isinstance(plot, str)
    if widget_id:
        assert widget_id in plot
