import base64
import colorsys
import html
from typing import Annotated, Any
from urllib.parse import quote

import pandas as pd
from pydantic import Field
from pydantic.json_schema import SkipJsonSchema
from wt_registry import register

from ecoscope.platform.annotations import (
    AdvancedField,
    DataFrame,
    JsonSerializableDataFrameModel,
)
from ecoscope.platform.tasks.results._chart import _resolve_palette_colors

# Categorical slots, most frequent event type first: the first 8 are a validated set
# (lightness band, colour-blind separation, contrast); the next 8 add hues between them.
# Types beyond 16 get generated hues (_extra_color), so every type has its own colour.
TYPE_COLORS = [
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
    "#eda100",
    "#e87ba4",
    "#008300",
    "#6250d6",
    "#e34948",
    "#0f9fb5",
    "#8c5a3c",
    "#a6b51b",
    "#b14fc5",
    "#1f4e8c",
    "#7cc4f0",
    "#9b2f5f",
    "#d4a3e8",
]
FONT = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif"
INK, MUTED, LINE, ACCENT = "#1f2933", "#6b7785", "#e6e9ee", "#3d5a80"
CARD_COLOR = "#f9f9f9"  # Ecoscope Desktop's widget card: fills that hide what scrolls under them

BASE_CSS = (
    f":root {{ --font:{FONT}; --ink:{INK}; --muted:{MUTED}; --line:{LINE}; --accent:{ACCENT}; --card:{CARD_COLOR}; }}"
    + """
* { box-sizing: border-box; }
html { height: 100%; overflow: hidden; }  /* body scrolls inside the widget, never taller than its frame */
body { height: 100%; overflow: auto; font-family: var(--font); margin: 0; padding: 8px 16px;
       color: var(--ink); background: transparent; }  /* blend into the dashboard card */
.inc-id { display: inline-block; font-weight: 700; font-size: 13px; color: var(--accent); background: #eef2f7;
          border-radius: 6px; padding: 3px 8px; font-variant-numeric: tabular-nums; }
"""
)
STORYBOARD_CSS = (
    BASE_CSS
    + """
.row { display: grid; grid-template-columns: 150px 1fr; gap: 12px; align-items: center; padding: 8px 0;
       border-bottom: 1px solid #eef0f3; } .row:last-child { border-bottom: 0; }
.meta { color: var(--muted); font-size: 12px; margin-top: 8px; }
.ttl { font-size: 13px; margin-top: 6px; color: #4a5560; }
.scroll { overflow-x: auto; padding-bottom: 4px; }
.board { display: flex; align-items: flex-start; width: max-content; padding: 10px 0 4px; }
.tile { width: 116px; text-align: center; }
.ico { position: relative; width: 56px; height: 56px; margin: 0 auto; border-radius: 14px; background: #fff;
       border: 2px solid var(--c); display: flex; align-items: center; justify-content: center; }
.ico i { width: 22px; height: 22px; border-radius: 50%; background-color: var(--c); }
.ico i.ic { width: 34px; height: 34px; border-radius: 0; background-color: transparent; }
.ico b { position: absolute; top: -8px; right: -10px; background: #1f2933; color: #fff; font-size: 11px;
         border-radius: 999px; padding: 1px 6px; }
.ty { font-size: 12px; font-weight: 600; margin-top: 7px; white-space: nowrap; overflow: hidden;
      text-overflow: ellipsis; padding: 0 4px; }
.ids, .when { font-size: 11px; color: var(--muted); margin-top: 2px; line-height: 1.35;
              font-variant-numeric: tabular-nums; }
.nl { font-size: 10.5px; color: #9aa5b1; font-style: italic; margin-top: 2px; }
.arrow { width: 64px; position: relative; height: 60px; margin: 0 -14px; }
.arrow::before { content: ''; position: absolute; left: 4px; right: 10px; top: 29px; height: 2px; background: #c3cad4; }
.arrow::after { content: ''; position: absolute; right: 4px; top: 25px; border: 5px solid transparent;
                border-left: 7px solid #c3cad4; }
.arrow span { position: absolute; left: 50%; top: 8px; transform: translateX(-50%); font-size: 10.5px; color: #4a5560;
              background: var(--card); border: 1px solid #e1e6ed; border-radius: 999px; padding: 0 6px;
              white-space: nowrap; }
.ic { display: inline-block; background-position: center; background-size: contain; background-repeat: no-repeat; }
"""
)


def _page(css: str, body: str) -> str:
    return f"<!doctype html><html><head><meta charset='utf-8'><style>{css}</style></head><body>{body}</body></html>"


def _empty(message: str = "No incidents in this period.") -> str:
    css = f"body{{font:14px {FONT};color:{MUTED};padding:16px;background:transparent}}"
    return _page(css, f"<p>{html.escape(message)}</p>")


def _duration(seconds: float) -> str:
    if pd.isna(seconds) or seconds < 60:
        return "same time"
    for unit, n in (("day", 86400), ("h", 3600), ("min", 60)):
        if seconds >= n:
            v = round(seconds / n)
            return f"{v} {unit}{'s' if unit == 'day' and v != 1 else ''}"
    return ""


def _located(geom: Any) -> bool:
    return geom is not None and not getattr(geom, "is_empty", True)


def _own_icon(uri: Any) -> str:
    """The icon, or "" when there is none or it is EarthRanger's generic icon (a plain disc
    given to types without their own), which is drawn as a dot of the type's colour instead."""
    u = uri if isinstance(uri, str) else ""
    text = u
    if u.startswith("data:image/svg+xml;base64,"):
        try:
            text = base64.b64decode(u.split(",", 1)[1]).decode(errors="ignore")
        except ValueError:
            return u
    if "<title>generic" in text or not u.startswith(("data:", "https://", "http://")):
        return ""
    return quote(u, safe=":/?#[]@!$&*+,;=%-._~")  # nothing that can close the CSS url('...') it goes in


def _css_color(c: Any) -> str | None:
    if c is None or (isinstance(c, float) and pd.isna(c)):
        return None
    if isinstance(c, str):
        return c
    r, g, b, a = (list(c) + [255])[:4]
    return f"rgba({int(r)}, {int(g)}, {int(b)}, {int(a) / 255})"


def _extra_color(i: int) -> str:
    """Colour for the 17th+ event type: golden-angle hues, alternating lightness."""
    h = (i * 137.508 + 20) % 360 / 360
    r, g, b = colorsys.hls_to_rgb(h, 0.42 if i % 2 else 0.58, 0.55)
    return f"#{round(r * 255):02x}{round(g * 255):02x}{round(b * 255):02x}"


def _type_colors(types: pd.Series, palette: str | list[str] | None) -> dict[Any, str]:
    """One colour per type, the most frequent first (ties by name)."""
    counts = types.value_counts()
    order = sorted(counts.index, key=lambda t: (-counts[t], str(t)))
    colors = _resolve_palette_colors(palette, len(order))
    if colors:
        return {t: colors[i % len(colors)] for i, t in enumerate(order)}
    n = len(TYPE_COLORS)
    return {t: TYPE_COLORS[i] if i < n else _extra_color(i - n) for i, t in enumerate(order)}


def _board(g: pd.DataFrame, merge_s: float, icon_classes: dict[str, str]) -> str:
    esc = html.escape
    runs: list[list[Any]] = []
    for e in g.itertuples():
        last = runs[-1][-1] if runs else None
        if last is not None and last.sb_type == e.sb_type and (e.sb_time - last.sb_time).total_seconds() < merge_s:
            runs[-1].append(e)
        else:
            runs.append([e])
    out = []
    for k, run in enumerate(runs):
        first, last = run[0], run[-1]
        if k:
            gap = (first.sb_time - runs[k - 1][-1].sb_time).total_seconds()
            out.append(f"<div class='arrow'><span>{_duration(gap)}</span></div>")
        icon = f"<i class='ic {icon_classes[first.sb_icon]}'></i>" if first.sb_icon else "<i></i>"
        n = len(run)
        when = f"{first.sb_time.day} {first.sb_time:%b %Y}<br>{first.sb_time:%H:%M}"
        if n > 1:
            when += f"–{last.sb_time:%H:%M}"
        ids = esc(str(first.sb_id)) if n == 1 else f"{esc(str(first.sb_id))} … {esc(str(last.sb_id))}"
        unlocated = sum(not e.sb_located for e in run)
        nl = ""
        if unlocated:
            nl = f"<div class='nl'>{'no location' if n == 1 else f'{unlocated} without location'}</div>"
        badge = f"<b>×{n}</b>" if n > 1 else ""
        ty = esc(str(first.sb_type))
        out.append(
            f"<div class='tile' title='{ty}' style='--c:{esc(first.sb_color)}'>"
            f"<div class='ico'>{icon}{badge}</div>"
            f"<div class='ty'>{ty}</div><div class='ids'>{ids}</div><div class='when'>{when}</div>{nl}</div>"
        )
    return f"<div class='board'>{''.join(out)}</div>"


def _label(group: Any) -> str:
    """'#<serial> <title>' as an id chip with the title under it; any other label as the chip."""
    text = str(group)
    sid, _, title = text.partition(" ") if text.startswith("#") else (text, "", "")
    chip = f"<span class='inc-id'>{html.escape(sid)}</span>"
    return chip + (f"<div class='ttl'>{html.escape(title)}</div>" if title else "")


@register()
def draw_incident_storyboard(
    dataframe: DataFrame[JsonSerializableDataFrameModel],
    group_column: Annotated[
        str, AdvancedField(default="incident", description="One storyboard row per value of this column.")
    ] = "incident",
    group_time_column: Annotated[
        str,
        AdvancedField(
            default="incident_time",
            description="Orders the rows, newest first. If the column is missing, each row's first event time is used.",
        ),
    ] = "incident_time",
    time_column: Annotated[str, AdvancedField(default="time", description="The event time.")] = "time",
    type_column: Annotated[
        str,
        AdvancedField(default="event_type_display", description="The event type: tile label, merge and colour key."),
    ] = "event_type_display",
    id_column: Annotated[
        str, AdvancedField(default="serial_number", description="The event id shown on each tile.")
    ] = "serial_number",
    icon_column: Annotated[
        str,
        AdvancedField(
            default="icon",
            description="Event icons (data URI or URL). "
            "A missing column or empty value draws a dot of the type's colour.",
        ),
    ] = "icon",
    color_column: Annotated[
        str | SkipJsonSchema[None],
        AdvancedField(
            default=None,
            description="Per-event colours (css or RGBA tuples), e.g. to keep colours the same across dashboard views. "
            "Overrides palette.",
        ),
    ] = None,
    palette: Annotated[
        str | list[str] | SkipJsonSchema[None],
        AdvancedField(
            default=None,
            description="Event type colours, most frequent type first: "
            "a named matplotlib colormap or a list of colours.",
        ),
    ] = None,
    merge_window_hours: Annotated[
        float,
        AdvancedField(default=12, ge=0, description="Same-type events this close share one tile (×N). 0 never merges."),
    ] = 12,
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
    Draws incident storyboards: one row per incident, newest first, with its events in
    time order as tiles (icon framed in the event type's colour, type, id, date and time).
    Arrows carry the time between tiles; same-type events within the merge window share
    one tile (×N); events without a location say so; long rows scroll sideways.

    Args:
    dataframe (pd.DataFrame): Events, one row per (incident, event), e.g. from get_incident_events.
    group_column (str): One storyboard row per value of this column.
    group_time_column (str): Orders the rows, newest first.
    time_column (str): The event time.
    type_column (str): The event type.
    id_column (str): The event id shown on each tile.
    icon_column (str): Event icons (data URI or URL).
    color_column (str): Per-event colours; overrides palette.
    palette (str | list[str]): Event type colours, most frequent type first.
    merge_window_hours (float): Same-type events this close share one tile; 0 never merges.
    widget_id (str): The id of the dashboard widget that this tile layer belongs to.

    Returns:
    The storyboards as an HTML page
    """
    if dataframe.empty:
        return _empty()
    df = dataframe.copy()
    if "id" in df.columns:
        df = df.drop_duplicates([group_column, "id"])
    df["sb_time"] = pd.to_datetime(df[time_column], utc=True, format="ISO8601")
    df = df[df["sb_time"].notna()]  # an event without a time can't be placed on the board
    if df.empty:
        return _empty()
    df["sb_type"] = df[type_column].fillna("(no type)")
    df["sb_id"] = df[id_column] if id_column in df.columns else ""
    df["sb_icon"] = df[icon_column].map(_own_icon) if icon_column in df.columns else ""
    geometry = df.geometry if "geometry" in df.columns else pd.Series(True, index=df.index)
    df["sb_located"] = [g is True or _located(g) for g in geometry]
    events = df.drop_duplicates("id") if "id" in df.columns else df
    df["sb_color"] = df["sb_type"].map(_type_colors(events["sb_type"], palette))
    if color_column:  # own colours win; a missing one falls back to the palette's
        df["sb_color"] = df[color_column].map(_css_color).fillna(df["sb_color"])
    sort_ids = ["sb_time", id_column] if id_column in df.columns else ["sb_time"]
    df = df.sort_values(sort_ids, kind="stable")

    icon_classes = {u: f"ic{i}" for i, u in enumerate(dict.fromkeys(u for u in df["sb_icon"] if u))}
    icon_css = "".join(f".{c}{{background-image:url('{u}')}}" for u, c in icon_classes.items())

    def started(g: pd.DataFrame) -> pd.Timestamp:
        if group_time_column in g.columns:
            return pd.to_datetime(g[group_time_column], utc=True, format="ISO8601").iloc[0]
        return g["sb_time"].iloc[0]

    groups = sorted(df.groupby(group_column, sort=False), key=lambda kv: started(kv[1]), reverse=True)
    rows = []
    for group, g in groups:
        span = (g["sb_time"].max() - g["sb_time"].min()).total_seconds()
        meta = f"{len(g)} event{'s' if len(g) != 1 else ''}" + (f" · {_duration(span)}" if len(g) > 1 else "")
        rows.append(
            f"<div class='row'><div>{_label(group)}<div class='meta'>{meta}</div></div>"
            f"<div class='scroll'>{_board(g, merge_window_hours * 3600, icon_classes)}</div></div>"
        )
    body = "".join(rows)
    if widget_id:
        body = f"<div id='{html.escape(widget_id)}'>{body}</div>"
    return _page(STORYBOARD_CSS + icon_css, body)
