import importlib.resources
import json
from typing import Annotated

import pandas as pd
from pydantic import BaseModel, Field
from pydantic.json_schema import SkipJsonSchema
from wt_registry import register

from ecoscope.platform.annotations import AdvancedField, AnyDataFrame

HTML_TEMPLATE = importlib.resources.read_text(__package__, "table-template.jinja2")


def _convert_json_columns_to_string(dataframe: AnyDataFrame) -> AnyDataFrame:
    for col in dataframe.columns:
        if dataframe[col].map(lambda x: isinstance(x, dict) or isinstance(x, list)).any():
            dataframe[col] = dataframe[col].map(lambda x: json.dumps(x))
    return dataframe


class TableConfig(BaseModel):
    enable_sorting: bool = True
    enable_filtering: bool = False
    enable_download: bool = False
    hide_header: bool = False


@register()
def draw_table(
    dataframe: Annotated[
        AnyDataFrame,
        Field(description="The dataframe to render as a table.", exclude=True),
    ],
    columns: Annotated[
        list[str] | SkipJsonSchema[None],
        AdvancedField(
            description="The list of dataframe columns to render in the table. Leave empty to render all columns",
            default=None,
        ),
    ] = None,
    table_config: Annotated[
        TableConfig | SkipJsonSchema[None],
        AdvancedField(
            description="Configuration options for the table.",
            default=None,
        ),
    ] = None,
    widget_id: Annotated[
        str | SkipJsonSchema[None],
        Field(
            description="""\
            The id of the dashboard widget that this table belongs to.
            If set this MUST match the widget title as defined downstream in create_widget tasks
            """,
            exclude=True,
        ),
    ] = None,
) -> Annotated[str, Field()]:
    """
    Creates an HTML table of the provided dataframe.
    """
    import geopandas as gpd  # type: ignore[import-untyped]

    if table_config is None:
        table_config = TableConfig()

    if isinstance(dataframe, gpd.GeoDataFrame):
        dataframe = dataframe.drop(columns="geometry")
        dataframe = pd.DataFrame(dataframe)  # type: ignore[assignment]

    dataframe = dataframe[columns] if columns else dataframe
    dataframe = _convert_json_columns_to_string(dataframe)

    table_row_data = dataframe.to_json(orient="records", date_format="iso")
    table_column_defs = json.dumps([{"field": col, "headerTooltip": col} for col in dataframe.columns])

    return HTML_TEMPLATE.format(
        table_row_data=table_row_data,
        table_column_defs=table_column_defs,
        table_config=table_config.model_dump_json(),
        widget_id=widget_id,
    )
