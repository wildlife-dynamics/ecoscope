import pathlib
from typing import Callable

import pandas as pd
import pandera as pa
import pytest
from pandera.typing import Series as PanderaSeries
from pydantic import ValidationError, validate_call

import ecoscope.distributed.types as edt


@pytest.fixture
def df_with_parquet_path(tmp_path) -> tuple[pd.DataFrame, str, pa.DataFrameModel]:
    path: pathlib.Path = tmp_path / "df.parquet"
    df = pd.DataFrame(data={'col1': [1, 2], 'col2': [3, 4]})
    df.to_parquet(path)

    class Schema(edt.JsonSerializableDataFrameModel):
        col1: PanderaSeries[int] = pa.Field(unique=True)
        col2: PanderaSeries[int] = pa.Field(unique=True)

    return df, path.as_posix(), Schema


def test_InputDataframe_coercion(df_with_parquet_path: tuple[pd.DataFrame, str, pa.DataFrameModel]):
    df, parquet_path, Schema = df_with_parquet_path

    def get_df(df: edt.DataFrame[Schema]):  # type: ignore
        return df

    # without coercion: pass a df, get a df; pass a str, get a str
    df_passthrough = get_df(df)
    assert isinstance(df_passthrough, pd.DataFrame)
    df_from_path_no_coercion = get_df(parquet_path)
    assert isinstance(df_from_path_no_coercion, str)

    # with coercion: pass a path to a parquet file, get a df back
    get_df_with_coercion: Callable = validate_call(
        get_df, config={"arbitrary_types_allowed": True},
    )
    df_from_path_with_coercion = get_df_with_coercion(parquet_path)
    assert isinstance(df_from_path_with_coercion, pd.DataFrame)

    # and it round-trips
    pd.testing.assert_frame_equal(df, df_from_path_with_coercion)


def test_InputDataframe_schema_validation_passes(df_with_parquet_path):
    df, parquet_path, Schema = df_with_parquet_path    

    def get_df(df: edt.DataFrame[Schema]):  # type: ignore
        return df

    get_df_with_coercion: Callable = validate_call(
        get_df, config={"arbitrary_types_allowed": True},
    )
    df_from_path_with_coercion = get_df_with_coercion(parquet_path)
    assert isinstance(df_from_path_with_coercion, pd.DataFrame)
    pd.testing.assert_frame_equal(df, df_from_path_with_coercion)


def test_InputDataframe_schema_validation_fails(df_with_parquet_path):
    _, parquet_path, _ = df_with_parquet_path

    class IncorrectSchema(edt.JsonSerializableDataFrameModel):
        col1: PanderaSeries[str] = pa.Field(unique=True)

    def get_df(df: edt.DataFrame[IncorrectSchema]):
        return df

    get_df_with_coercion: Callable = validate_call(
        get_df, config={"arbitrary_types_allowed": True},
    )
    with pytest.raises(ValidationError):
        _ = get_df_with_coercion(parquet_path)


# TODO: Results model -> i.e. **validate/serialize return values**
# TODO: hardware/metal
# TODO: importable with minimal dependencies
# TODO: notebook example
# TODO: double decorator for validate call? e.g. `@checkpoint/@cache`
