from typing import Annotated

import pandas as pd  # FIXME

try:
    ...
    from pydantic import Field
    from pydantic.functional_validators import AfterValidator, BeforeValidator
except ImportError:
    Field = dict
    BeforeValidator = tuple


def maybe_load_dataframe_from_parquet_url(url_or_table: pd.DataFrame | str):
    # import geopandas as gpd

    return (
        url_or_table
        if isinstance(url_or_table, pd.DataFrame)
        else pd.read_parquet(url_or_table)
    )


Dataframe = Annotated[
    pd.DataFrame | str,
    BeforeValidator(maybe_load_dataframe_from_parquet_url)
]
