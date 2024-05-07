from typing import Annotated, TypeVar

import pandas as pd
import pandera as pa
from pydantic_core import core_schema as cs
from pandera.typing import DataFrame as PanderaDataFrame
from pydantic import GetJsonSchemaHandler
from pydantic.functional_validators import AfterValidator, BeforeValidator
from pydantic.json_schema import JsonSchemaValue, SkipJsonSchema


class DataFrameModel(pa.DataFrameModel):
    @classmethod
    def __get_pydantic_json_schema__(
        cls, core_schema: cs.CoreSchema, handler: GetJsonSchemaHandler
    ) -> JsonSchemaValue:
        return cls.to_json_schema()


class Deserializer(BeforeValidator):
    pass
    # @classmethod
    # def __get_pydantic_json_schema__(
    #     cls, core_schema: cs.CoreSchema, handler: GetJsonSchemaHandler
    # ) -> JsonSchemaValue:
    #     return {}


def load_dataframe_from_parquet_url(url: str):
    # TODO: geopandas read_parquet
    return pd.read_parquet(url)


DataframeSchema = TypeVar("DataframeSchema", bound=DataFrameModel)

InputDataframe = Annotated[
    PanderaDataFrame[DataframeSchema],
    Deserializer(load_dataframe_from_parquet_url),
]


def persist_dataframe(df: pd.DataFrame):
    # persist dataframe here
    url = ...
    return url


OutputDataframe = Annotated[
    PanderaDataFrame[DataframeSchema],
    AfterValidator(persist_dataframe)
]
