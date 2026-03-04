import geopandas as gpd  # type: ignore[import-untyped]
import numpy as np
import pandas as pd
from ecoscope.platform.tasks.results._table import TableConfig, draw_table
from shapely.geometry import Point


def test_draw_table():
    df = pd.DataFrame(
        {
            "Some Numbers": [1, 2, 3],
            "More Numbers!": [4, 5, 6],
        }
    )

    expected_column_data = 'const columnDefs = [{"field": "Some Numbers", "headerTooltip": "Some Numbers"}, {"field": "More Numbers!", "headerTooltip": "More Numbers!"}]'
    expected_row_data = 'const rowData = [{"Some Numbers":1,"More Numbers!":4},{"Some Numbers":2,"More Numbers!":5},{"Some Numbers":3,"More Numbers!":6}]'

    html = draw_table(df)
    assert isinstance(html, str)
    assert expected_column_data in html
    assert expected_row_data in html


def test_draw_table_column_filter():
    df = pd.DataFrame(
        {
            "Some Numbers": [1, 2, 3],
            "More Numbers!": [4, 5, 6],
            "Irrelevant column": ["please", "ignore", "me"],
        }
    )

    html = draw_table(df, columns=["Some Numbers", "More Numbers!"])
    assert isinstance(html, str)
    assert "Some Numbers" in html
    assert "More Numbers!" in html
    assert "Irrelevant column" not in html


def test_draw_table_with_widget_id():
    df = pd.DataFrame(
        {
            "Some Numbers": [1, 2, 3],
            "More Numbers!": [4, 5, 6],
        }
    )

    widget_id = "very important widget id"
    html = draw_table(df, widget_id=widget_id)
    assert isinstance(html, str)
    assert "Some Numbers" in html
    assert "More Numbers!" in html
    assert f'widgetId: "{widget_id}"' in html


def test_draw_table_json_column():
    df = pd.DataFrame(
        {
            "Some Numbers": [1, 2, 3],
            "JSON Column": [
                {"nested": "stuff"},
                {"even": {"more": "nested"}},
                {
                    "this": {
                        "one": ["has", "an", "array"],
                    }
                },
            ],
        }
    )

    html = draw_table(df)
    assert isinstance(html, str)
    assert "Some Numbers" in html
    assert "JSON Column" in html
    assert '{\\"nested\\": \\"stuff\\"}' in html
    assert '{\\"even\\": {\\"more\\": \\"nested\\"}}' in html
    assert '{\\"this\\": {\\"one\\": [\\"has\\", \\"an\\", \\"array\\"]}}' in html


def test_draw_table_timestamp_format():
    df = pd.DataFrame(
        {
            "Some Numbers": [1, 2, 3],
            "More Numbers!": [4, 5, 6],
            "Timestamp": pd.to_datetime(
                ["2024-01-01", "2025-01-01", "2026-01-01"], utc=True
            ),
        }
    )

    html = draw_table(df)
    assert isinstance(html, str)
    assert "Some Numbers" in html
    assert "More Numbers!" in html
    assert "2024-01-01T00:00:00.000Z" in html
    assert "2025-01-01T00:00:00.000Z" in html
    assert "2026-01-01T00:00:00.000Z" in html


def test_draw_table_from_gdf():
    gdf = gpd.GeoDataFrame(
        {
            "id": ["A", "B"],
            "time": pd.to_datetime(["2024-01-01", "2025-01-01"], utc=True),
            "geometry": [Point(0.0, 0.0), Point(100.0, 50.0)],
        },
    )

    html = draw_table(gdf)
    assert isinstance(html, str)
    assert "geometry" not in html


def test_draw_table_handles_nan():
    df = pd.DataFrame(
        {
            "Some Numbers": [1, 2, 3],
            "More Numbers!": [4, np.nan, 6],
        }
    )
    expected_column_data = 'const columnDefs = [{"field": "Some Numbers", "headerTooltip": "Some Numbers"}, {"field": "More Numbers!", "headerTooltip": "More Numbers!"}]'
    # A quirk of pandas is that the presence of nan in a column of ints will cause a float coercion
    expected_row_data = 'const rowData = [{"Some Numbers":1,"More Numbers!":4.0},{"Some Numbers":2,"More Numbers!":null},{"Some Numbers":3,"More Numbers!":6.0}]'

    html = draw_table(df)
    assert isinstance(html, str)
    assert expected_column_data in html
    assert expected_row_data in html


def test_table_config():
    df = pd.DataFrame(
        {
            "Some Numbers": [1, 2, 3],
            "More Numbers!": [4, np.nan, 6],
        }
    )
    expected_config = 'const config = {"enable_sorting":false,"enable_filtering":true,"enable_download":true,"hide_header":true}'

    html = draw_table(
        df,
        table_config=TableConfig(
            enable_sorting=False,
            enable_filtering=True,
            enable_download=True,
            hide_header=True,
        ),
    )
    assert isinstance(html, str)
    assert expected_config in html
