from typing import Annotated

import pandas as pd  # FIXME

try:
    from pydantic import Field
    from pydantic.functional_validators import AfterValidator, BeforeValidator
except ImportError:
    Field = dict
    BeforeValidator = tuple


def maybe_load_dataframe_from_parquet_url(url_or_table: pd.DataFrame | str):
    return (
        url_or_table
        if isinstance(url_or_table, pd.DataFrame)
        # TODO: geopandas read_parquet
        else pd.read_parquet(url_or_table)
    )


InputDataframe = Annotated[
    pd.DataFrame | str,
    BeforeValidator(maybe_load_dataframe_from_parquet_url)
]


def persist_dataframe(df: pd.DataFrame):
    # persist dataframe here
    url = ...
    return url


OutputDataframe = Annotated[
    pd.DataFrame,
    AfterValidator(persist_dataframe)
]
