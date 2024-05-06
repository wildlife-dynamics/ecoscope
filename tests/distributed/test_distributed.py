import pathlib
from typing import Callable

import pandas as pd
import pytest
from pydantic import validate_call

import ecoscope.distributed.types as edt


@pytest.fixture
def df_with_parquet_path(tmp_path) -> tuple[pd.DataFrame, str]:
    path: pathlib.Path = tmp_path / "df.parquet"
    df = pd.DataFrame(data={'col1': [1, 2], 'col2': [3, 4]})
    df.to_parquet(path)     
    return df, path.as_posix()


def get_df(df: edt.InputDataframe):
    return df


def test_dataframe_type(df_with_parquet_path: tuple[pd.DataFrame, str]):
    df, parquet_path = df_with_parquet_path

    # obviously if we pass through a df, we get a df
    df_passthrough = get_df(df)
    assert isinstance(df_passthrough, pd.DataFrame)
    
    # ...and without coercion, we get a str if we give a str
    df_from_path_no_coercion = get_df(parquet_path)
    assert isinstance(df_from_path_no_coercion, str)

    # but if we wrap `get_df` in pydantic validation
    get_df_with_coercion: Callable = validate_call(
        get_df,
        # let's pydantic accept pandas dataframe as type
        config={"arbitrary_types_allowed": True},
    )
    # ...and pass it a path to a parquet dataset...
    df_from_path_with_coercion = get_df_with_coercion(parquet_path)
    # then the `Dataframe` type auto-magically loads the df
    assert isinstance(df_from_path_with_coercion, pd.DataFrame)

    # and it round-trips
    pd.testing.assert_frame_equal(df, df_from_path_with_coercion)


# TODO: test BaseModel generation for frontend schema
# TODO: Results model
# TODO: hardware/metal
# TODO: pandera
# TODO: importable with minimal dependencies
