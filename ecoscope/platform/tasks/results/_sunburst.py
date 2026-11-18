from typing import Annotated, Literal

from pydantic import BaseModel, Field
from pydantic.json_schema import SkipJsonSchema
from wt_registry import register

from ecoscope.platform.annotations import (
    AdvancedField,
    CacheableModel,
    DataFrame,
    JsonSerializableDataFrameModel,
)
from ecoscope.platform.tasks.results._chart import _resolve_palette_colors
from ecoscope.platform.tasks.results._ecoplot import ExportArgs, LayoutStyle


class SunburstStyle(BaseModel, CacheableModel):
    """Style kwargs passed to go.Sunburst(); unset fields keep the chart's defaults."""

    maxdepth: Annotated[
        Annotated[int, Field(ge=1)] | SkipJsonSchema[None],
        AdvancedField(default=None, description="How many rings to show at once; click a slice to see deeper."),
    ] = None
    insidetextorientation: Annotated[
        Literal["horizontal", "radial", "tangential", "auto"] | SkipJsonSchema[None],
        AdvancedField(default=None, description="The direction of the text inside the slices."),
    ] = None
    rotation: Annotated[
        float | SkipJsonSchema[None],
        AdvancedField(default=None, description="The angle in degrees the first slice starts at."),
    ] = None
    sort: Annotated[
        bool | SkipJsonSchema[None],
        AdvancedField(default=None, description="Draw the largest slices first."),
    ] = None
    textinfo: Annotated[
        str | SkipJsonSchema[None],
        AdvancedField(
            default=None,
            description="What the slice text shows, e.g. 'label+percent parent' or 'label+value'.",
        ),
    ] = None
    texttemplate: Annotated[
        str | SkipJsonSchema[None],
        AdvancedField(default=None, description="A plotly template for the slice text; overrides textinfo."),
    ] = None
    hovertemplate: Annotated[
        str | SkipJsonSchema[None],
        AdvancedField(default=None, description="A plotly template for the hover text."),
    ] = None


@register()
def draw_sunburst_chart(
    dataframe: DataFrame[JsonSerializableDataFrameModel],
    ring_columns: Annotated[
        list[str],
        Field(min_length=1, description="The columns drawn as rings, from the inside out."),
    ],
    value_column: Annotated[
        str | SkipJsonSchema[None],
        AdvancedField(
            default=None,
            description="A numeric column summed for each slice's size. If not set, each row counts once.",
        ),
    ] = None,
    color_column: Annotated[
        str | SkipJsonSchema[None],
        AdvancedField(
            default=None,
            description="A column of colours for the inner ring's values, e.g. from apply_color_map. "
            "Outer slices take their inner ring's colour. Overrides palette.",
        ),
    ] = None,
    palette: Annotated[
        str | list[str] | SkipJsonSchema[None],
        AdvancedField(
            default=None,
            description="Inner ring colours: a named matplotlib colormap or a list of colours.",
        ),
    ] = None,
    top_n: Annotated[
        Annotated[int, Field(ge=1)] | SkipJsonSchema[None],
        AdvancedField(
            default=None,
            description='Keep the largest N slices within each parent and merge the rest into "Other".',
        ),
    ] = None,
    pinned_values: Annotated[
        list[str] | SkipJsonSchema[None],
        AdvancedField(
            default=None,
            description='Values never merged into "Other" and not counted towards top_n, e.g. "(blank)".',
        ),
    ] = None,
    other_color: Annotated[
        str | SkipJsonSchema[None],
        AdvancedField(default=None, description='The colour of "Other" slices. Defaults to grey.'),
    ] = None,
    expand_other: Annotated[
        bool,
        AdvancedField(
            default=False,
            description='Keep the merged values inside "Other", so clicking "Other" shows them by name.',
        ),
    ] = False,
    plot_style: Annotated[
        SunburstStyle | SkipJsonSchema[None],
        AdvancedField(default=None, description="Additional style kwargs passed to go.Sunburst()."),
    ] = None,
    layout_style: Annotated[
        LayoutStyle | SkipJsonSchema[None],
        AdvancedField(default=None, description="Additional kwargs passed to plotly.go.Figure(layout)."),
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
    Generates a sunburst chart: one ring per column, inside out, each slice sized by
    its row count or summed value_column. A missing value ends a row's path, so it
    adds no slice in that ring or any ring outside it.

    Args:
    dataframe (pd.DataFrame): The input dataframe, one row per record.
    ring_columns (list[str]): The columns drawn as rings, from the inside out.
    value_column (str): A numeric column summed for each slice's size.
    color_column (str): A column of colours for the inner ring's values; overrides palette.
    palette (str | list[str]): A named matplotlib colormap or a list of colours for the inner ring.
    top_n (int): Keep the largest N slices within each parent; merge the rest into "Other".
    pinned_values (list[str]): Values never merged into "Other".
    other_color (str): The colour of "Other" slices.
    expand_other (bool): Keep the merged values inside "Other" instead of dropping their names.
    plot_style (SunburstStyle): Additional style kwargs passed to go.Sunburst().
    layout_style (LayoutStyle): Additional kwargs passed to plotly.go.Figure(layout).
    widget_id (str): The id of the dashboard widget that this tile layer belongs to.
        If set this MUST match the widget title as defined downstream in create_widget tasks

    Returns:
    The generated chart html as a string
    """
    from ecoscope.plotting import sunburst_chart

    plot = sunburst_chart(
        data=dataframe,
        ring_columns=ring_columns,
        value_column=value_column,
        color_column=color_column,
        palette=_resolve_palette_colors(palette, dataframe[ring_columns[0]].nunique()),
        top_n=top_n,
        pinned_values=pinned_values,
        expand_other=expand_other,
        style_kwargs=plot_style.model_dump(exclude_none=True) if plot_style else {},
        layout_kwargs=layout_style.model_dump(exclude_none=True) if layout_style else {},
        **({"other_color": other_color} if other_color else {}),
    )

    return plot.to_html(**ExportArgs(div_id=widget_id).model_dump(exclude_none=True))
