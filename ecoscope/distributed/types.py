from typing import Annotated, TypeVar

import pandas as pd  # FIXME

try:
    from pydantic import Field
    from pydantic.functional_validators import AfterValidator, BeforeValidator
except ImportError:
    Field = dict
    BeforeValidator = tuple


def load_dataframe_from_parquet_url(url: str):
    # TODO: geopandas read_parquet
    return pd.read_parquet(url)


DataframeSchemaPlaceholder = TypeVar("DataframeSchemaPlaceholder")

InputDataframe = Annotated[
    DataframeSchemaPlaceholder,
    BeforeValidator(load_dataframe_from_parquet_url)
]


def persist_dataframe(df: pd.DataFrame):
    # persist dataframe here
    url = ...
    return url


OutputDataframe = Annotated[
    DataframeSchemaPlaceholder,
    AfterValidator(persist_dataframe)
]
