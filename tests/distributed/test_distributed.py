import pathlib

import pandas as pd
from pydantic import validate_call

from ecoscope.distributed.types import Dataframe


def get_df(df: Dataframe):
    return df


def test_dataframe_type(tmp_path):
    path: pathlib.Path = tmp_path / "df.parquet"

    df = pd.DataFrame(data={'col1': [1, 2], 'col2': [3, 4]})
    df.to_parquet(path)  

    # obviously if we pass through a df, we get a df
    df_fromdf_novalidate = get_df(df)
    assert isinstance(df_fromdf_novalidate, pd.DataFrame)
    
    # ...and without coercion, we get a str if we give a str
    df_fromstr_novalidate = get_df(path.as_posix())
    assert isinstance(df_fromstr_novalidate, str)

    # but if we wrap `get_df` in pydantic validation
    validated_get_df = validate_call(
        get_df,
        # let's pydantic accept pandas dataframe as type
        config={"arbitrary_types_allowed": True}
    )
    # ...and pass it a path to a parquet dataset...
    df_fromstr_validate = validated_get_df(path.as_posix())
    # then the `Dataframe` type auto-magically loads the df
    assert isinstance(df_fromstr_validate, pd.DataFrame)

    # and it round-trips
    pd.testing.assert_frame_equal(df, df_fromstr_validate)
