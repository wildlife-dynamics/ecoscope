import pandas as pd
import pandera as pa
import pytest
from pandera.typing import Series as PanderaSeries
from pydantic import BaseModel, ValidationError

from ecoscope.distributed.types import DataFrame, JsonSerializableDataFrameModel


def test_dataframe_type() -> tuple[pd.DataFrame, pa.DataFrameModel]:
    df = pd.DataFrame(data={'col1': [1, 2]})

    class ValidSchema(JsonSerializableDataFrameModel):
        col1: PanderaSeries[int] = pa.Field(unique=True)

    class ValidModel(BaseModel):
        df: DataFrame[ValidSchema]

    ValidModel(df=df)

    class InvalidSchema(JsonSerializableDataFrameModel):
        # Invalid because col1 elements are ints, not strings
        col1: PanderaSeries[str] = pa.Field(unique=True)

    class InvalidModel(BaseModel):
        df: DataFrame[InvalidSchema]

    with pytest.raises(ValidationError):
        InvalidModel(df=df)
