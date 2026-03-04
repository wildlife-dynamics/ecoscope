import json
import logging
from enum import Enum
from typing import Annotated, cast

import numpy as np
import pandas as pd
from pydantic import Field
from wt_registry import register

from ecoscope.platform.annotations import AnyDataFrame

logger = logging.getLogger(__name__)


class FieldType(Enum):
    STRING = "str"
    FLOAT = "float"
    BOOL = "bool"
    DATETIME = "datetime"
    DATE = "date"
    JSON = "json"
    SERIES = "series"

    @classmethod
    def from_string(cls, value: str):
        try:
            return cls(value.lower())
        except ValueError:
            raise ValueError(
                f"'{value}' is not a valid FieldType. Valid options are: {[e.value for e in cls]}"
            )


def extract_value_as_type(value, output_type: FieldType):
    if value is None:
        return None

    try:
        match output_type:
            case FieldType.STRING:
                return str(value)
            case FieldType.FLOAT:
                return np.float64(value)
            case FieldType.BOOL:
                value = str(value)
                if value.lower() in ("true", "1"):
                    return True
                elif value.lower() in ("false", "0"):
                    return False
                return None
            case FieldType.DATETIME:
                return pd.to_datetime(value, utc=True)
            case FieldType.DATE:
                return pd.to_datetime(value, utc=True).date()
            case FieldType.JSON:
                return json.dumps(value) if isinstance(value, (dict, list)) else value
            case FieldType.SERIES:
                return pd.Series(value)
    except (json.JSONDecodeError, TypeError, ValueError):
        return None

    return None


@register()
def extract_column_as_type(
    df: Annotated[
        AnyDataFrame,
        Field(
            description="The dataframe.",
            exclude=True,
        ),
    ],
    column_name: Annotated[
        str, Field(description="The column name to extract the value from.")
    ],
    output_type: Annotated[
        FieldType, Field(description="The output type of the extracted value.")
    ],
    output_column_name: Annotated[
        str,
        Field(
            description="The output column name to store the extracted value. If it's a pandas series, then the output_column_name will be the column prefix."
        ),
    ],
) -> AnyDataFrame:
    output = df[column_name].apply(lambda x: extract_value_as_type(x, output_type))
    if output_type == FieldType.SERIES:
        output_df = output.add_prefix(output_column_name)
        result_df = df.merge(output_df, right_index=True, left_index=True)
    else:
        df[output_column_name] = output
        result_df = df

    return cast(
        AnyDataFrame,
        result_df,
    )


@register()
def extract_value_from_json_column(
    df: Annotated[
        AnyDataFrame,
        Field(
            description="The dataframe.",
            exclude=True,
        ),
    ],
    column_name: Annotated[
        str, Field(description="The json column name to extract the value from.")
    ],
    field_name_options: Annotated[
        list[str],
        Field(
            description="A list of field name options to extract the value from. The first field name that is found will be used."
        ),
    ],
    output_type: Annotated[
        FieldType, Field(description="The output type of the extracted value.")
    ],
    output_column_name: Annotated[
        str,
        Field(
            description="The output column name to store the extracted value. If it's a pandas series, then the output_column_name will be the column prefix."
        ),
    ],
) -> AnyDataFrame:
    def extract_value_from_row(row):
        additional = row[column_name] or {}
        if isinstance(additional, str):
            additional = json.loads(additional)

        value = None
        for field in field_name_options:
            value = additional.get(field, None)
            if value is not None:
                break

        if value is None:
            return value

        return extract_value_as_type(value, output_type)

    output = df.apply(extract_value_from_row, axis=1)

    if output_type == FieldType.SERIES:
        output_df = output.add_prefix(output_column_name)
        result_df = df.merge(output_df, right_index=True, left_index=True)
    else:
        df[output_column_name] = output
        result_df = df

    return cast(
        AnyDataFrame,
        result_df,
    )
