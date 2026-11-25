import base64
import re
from importlib.resources import files

import geopandas as gpd  # type: ignore[import-untyped]
import pandas as pd
from shapely.geometry import Point

from ecoscope.platform.mock_loaders import load_parquet
from ecoscope.platform.tasks.results import draw_incident_storyboard

ICON = "data:image/svg+xml;base64," + base64.b64encode(b"<svg width='64' height='64'></svg>").decode()
GENERIC = "data:image/svg+xml;base64," + base64.b64encode(b"<svg><title>generic-gray</title></svg>").decode()


def _df(events, incidents=None):
    """events: (incident, time, type, serial, extra-dict) tuples."""
    incidents = incidents or {}
    rows = []
    for i, (inc, time, etype, serial, *extra) in enumerate(events):
        rows.append(
            {
                "incident": inc,
                "incident_time": incidents.get(inc, "2025-01-01T00:00:00Z"),
                "id": f"e{i}",
                "serial_number": serial,
                "time": time,
                "event_type_display": etype,
                "icon": "",
                "geometry": Point(36.8, -1.3),
                **(extra[0] if extra else {}),
            }
        )
    df = gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")
    for c in ("incident_time", "time"):
        df[c] = pd.to_datetime(df[c], utc=True)
    return df


def _tiles(html):
    return re.findall(r"<div class='tile'", html)


def test_rows_newest_incident_first():
    html = draw_incident_storyboard(
        _df(
            [("#1 Old", "2025-01-01T01:00:00Z", "Snare", 10), ("#2 New", "2025-02-01T01:00:00Z", "Snare", 11)],
            {"#1 Old": "2025-01-01T00:00:00Z", "#2 New": "2025-02-01T00:00:00Z"},
        )
    )

    assert html.index("New") < html.index("Old")
    assert html.count("class='row'") == 2


def test_same_type_within_window_merges():
    html = draw_incident_storyboard(_df([("#1 A", f"2025-01-01T0{h}:00:00Z", "Snare", 5 + h) for h in range(3)]))

    assert len(_tiles(html)) == 1
    assert "×3" in html
    assert "5 … 7" in html


def test_merge_window_and_type_break_runs():
    far = draw_incident_storyboard(
        _df([("#1 A", "2025-01-01T00:00:00Z", "Snare", 1), ("#1 A", "2025-01-01T13:00:00Z", "Snare", 2)])
    )
    between = draw_incident_storyboard(
        _df(
            [
                ("#1 A", "2025-01-01T00:00:00Z", "Snare", 1),
                ("#1 A", "2025-01-01T00:20:00Z", "Fire", 2),
                ("#1 A", "2025-01-01T00:40:00Z", "Snare", 3),
            ]
        )
    )
    never = draw_incident_storyboard(
        _df([("#1 A", "2025-01-01T00:00:00Z", "Snare", 1), ("#1 A", "2025-01-01T00:10:00Z", "Snare", 2)]),
        merge_window_hours=0,
    )

    assert len(_tiles(far)) == 2
    assert len(_tiles(between)) == 3
    assert len(_tiles(never)) == 2 and "×" not in never


def test_arrow_durations():
    html = draw_incident_storyboard(
        _df(
            [
                ("#1 A", "2025-01-01T00:00:00Z", "Snare", 1),
                ("#1 A", "2025-01-01T00:00:30Z", "Fire", 2),
                ("#1 A", "2025-01-01T02:00:30Z", "Gun", 3),
                ("#1 A", "2025-01-04T02:00:30Z", "Arrest", 4),
            ]
        )
    )

    assert re.findall(r"<div class='arrow'><span>([^<]*)</span>", html) == ["same time", "2 h", "3 days"]


def test_no_location_marked():
    df = _df([("#1 A", "2025-01-01T00:00:00Z", "Snare", 1)])
    df.loc[0, "geometry"] = None

    assert "no location" in draw_incident_storyboard(df)


def test_icon_or_dot():
    with_icon = _df(
        [
            ("#1 A", "2025-01-01T00:00:00Z", "Snare", 1, {"icon": ICON}),
            ("#1 A", "2025-01-02T00:00:00Z", "Snare", 2, {"icon": ICON}),
        ]
    )
    html = draw_incident_storyboard(with_icon)
    no_column = draw_incident_storyboard(_df([("#1 A", "2025-01-01T00:00:00Z", "Snare", 1)]).drop(columns="icon"))
    generic = draw_incident_storyboard(_df([("#1 A", "2025-01-01T00:00:00Z", "Snare", 1, {"icon": GENERIC})]))

    assert html.count(ICON) == 1
    assert "<i class='ic " in html
    assert "<i class='ic " not in no_column
    assert "<i class='ic " not in generic and GENERIC not in generic


def test_color_precedence():
    events = [
        ("#1 A", "2025-01-01T00:00:00Z", "Snare", 1),
        ("#1 A", "2025-01-01T00:10:00Z", "Snare", 2),
        ("#1 A", "2025-01-02T00:00:00Z", "Fire", 3),
    ]
    default = draw_incident_storyboard(_df(events), merge_window_hours=0)
    own = draw_incident_storyboard(
        _df([(*e, {"type_color": "#123456"}) for e in events]), color_column="type_color", palette=["#ffffff"]
    )
    tuples = draw_incident_storyboard(_df([(*e, {"c": (255, 0, 0, 255)}) for e in events]), color_column="c")

    assert "--c:#2a78d6" in default  # most frequent type (Snare) takes the first default colour
    assert "--c:#eb6834" in default
    assert "--c:#123456" in own and "#ffffff" not in own
    assert "--c:rgba(255, 0, 0, 1.0)" in tuples


def test_palette_list():
    html = draw_incident_storyboard(_df([("#1 A", "2025-01-01T00:00:00Z", "Snare", 1)]), palette=["#00aa00"])

    assert "--c:#00aa00" in html


def test_escapes_html():
    html = draw_incident_storyboard(_df([("#1 <b>&'x", "2025-01-01T00:00:00Z", "<i>type", 1)]))

    assert "&lt;b&gt;&amp;" in html
    assert "<b>&'x" not in html and "<i>type" not in html


def test_times_in_the_columns_timezone():
    df = _df([("#1 A", "2025-01-01T08:09:00Z", "Snare", 1)])
    df["time"] = df["time"].dt.tz_convert("Africa/Nairobi")

    html = draw_incident_storyboard(df)

    assert "11:09" in html and "08:09" not in html


def test_iso_string_times():
    df = _df([("#1 A", "2025-01-01T05:00:00Z", "Fire", 2), ("#1 A", "2025-01-01T01:00:00Z", "Snare", 1)])
    as_text = df.assign(time=df["time"].dt.strftime("%Y-%m-%dT%H:%M:%SZ"), incident_time="2025-01-01T00:00:00Z")

    assert draw_incident_storyboard(as_text) == draw_incident_storyboard(df)
    assert draw_incident_storyboard(df).index("Snare") < draw_incident_storyboard(df).index("Fire")


def test_without_group_time_orders_by_first_event():
    df = _df([("#1 Old", "2025-01-01T01:00:00Z", "Snare", 10), ("#2 New", "2025-02-01T01:00:00Z", "Snare", 11)])
    html = draw_incident_storyboard(df.drop(columns="incident_time"))

    assert html.index("New") < html.index("Old")


def test_empty_dataframe():
    html = draw_incident_storyboard(pd.DataFrame(columns=["incident", "time"]))
    assert "No incidents" in html
    assert "class='snap'" not in html  # nothing to download


def test_png_download_button():
    html = draw_incident_storyboard(_df([("#1 A", "2025-01-01T00:00:00Z", "Snare", 1)]))

    assert html.count("<button class='snap'") == 1
    assert "a.download = 'incident-storyboard.png'" in html
    assert html.index("<script>") > html.index("class='row'")  # after the rows it measures


def test_widget_id():
    assert "storyboard-widget" in draw_incident_storyboard(
        _df([("#1 A", "2025-01-01T00:00:00Z", "Snare", 1)]), widget_id="storyboard-widget"
    )


def test_works_on_example_return():
    df = load_parquet(files("ecoscope.platform.tasks.io") / "get-incident-events.example-return.parquet")
    html = draw_incident_storyboard(df)

    assert html.count("class='row'") == df["incident"].nunique()
    assert "no location" in html


def test_icon_values_cannot_break_out_of_css():
    evil = "http://x/a.svg');}</style><script>alert(1)</script>"
    html = draw_incident_storyboard(
        _df(
            [
                ("#1 A", "2025-01-01T00:00:00Z", "Snare", 1, {"icon": evil}),
                ("#1 A", "2025-01-02T00:00:00Z", "Fire", 2, {"icon": "javascript:alert(1)"}),
                ("#1 A", "2025-01-03T00:00:00Z", "Gun", 3, {"icon": "https://er.example/it's (1).svg"}),
            ]
        )
    )

    assert "alert(1)" not in html
    assert "javascript:" not in html
    assert "url('https://er.example/it%27s%20%281%29.svg')" in html


def test_missing_values_still_draw():
    df = _df(
        [
            ("#1 A", "2025-01-01T00:00:00Z", None, 1, {"c": None}),
            ("#1 A", None, "Snare", 2, {"c": "#123456"}),
            ("#1 A", "2025-01-02T00:00:00Z", "Snare", 3, {"c": "#123456"}),
        ]
    )
    html = draw_incident_storyboard(df, color_column="c")

    assert "(no type)" in html
    assert "--c:#2a78d6" in html or "--c:#eb6834" in html  # the null colour falls back to the palette
    assert len(_tiles(html)) == 2  # the event without a time can't be placed and is left out


def test_dates_without_platform_specific_format():
    import inspect

    from ecoscope.platform.tasks.results import _storyboard

    assert "%-d" not in inspect.getsource(_storyboard)  # unsupported by strftime on Windows
    assert "1 Jan 2025" in draw_incident_storyboard(_df([("#1 A", "2025-01-01T05:00:00Z", "Snare", 1)]))


def test_whole_tile_shows_full_type_on_hover():
    name = "Invasive Plant Clearance Follow-up Survey"
    html = draw_incident_storyboard(_df([("#1 A", "2025-01-01T00:00:00Z", name, 1)]))

    assert re.search(rf"<div class='tile' title='{name}'", html)


def test_more_types_than_default_colours():
    from ecoscope.platform.tasks.results._storyboard import TYPE_COLORS

    events = [("#1 A", f"2025-01-{d + 1:02d}T00:00:00Z", f"Type {d:02d}", d) for d in range(18)]
    colors = re.findall(r"--c:(#[0-9a-f]{6})", draw_incident_storyboard(_df(events)))

    assert len(set(colors)) == 18
    assert colors[16] not in TYPE_COLORS and colors[17] not in TYPE_COLORS


def test_undecodable_icon_data_is_kept_as_given():
    bad = "data:image/svg+xml;base64,abc"  # bad padding: not decodable
    html = draw_incident_storyboard(_df([("#1 A", "2025-01-01T00:00:00Z", "Snare", 1, {"icon": bad})]))

    assert "url('data:image/svg+xml;base64,abc')" in html


def test_all_events_without_time():
    df = _df([("#1 A", None, "Snare", 1), ("#1 A", None, "Fire", 2)])

    assert "No incidents" in draw_incident_storyboard(df)


def test_colors_from_color_map_by_frequency():
    from ecoscope.platform.tasks.transformation import apply_color_map, color_map_by_frequency

    events = [
        ("#1 A", "2025-01-01T00:00:00Z", "Fire", 1),
        ("#1 A", "2025-01-03T00:00:00Z", "Snare", 2),
        ("#1 A", "2025-01-05T00:00:00Z", "Snare", 3),
    ]
    df = _df(events)
    colormap = color_map_by_frequency(df, "event_type_display", ["#2a78d6", "#eb6834"])
    df = apply_color_map(df, "event_type_display", colormap, output_column_name="event_type_color")
    html = draw_incident_storyboard(df, color_column="event_type_color")

    assert "--c:rgba(42, 120, 214, 1.0)" in html  # Snare, the most common, takes the first color
    assert "--c:rgba(235, 104, 52, 1.0)" in html
