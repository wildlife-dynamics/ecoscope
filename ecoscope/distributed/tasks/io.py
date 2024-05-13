from typing import Annotated, Any

import pandas as pd
import pandera as pa
from pydantic import Field

from ecoscope.distributed.decorators import distributed
from ecoscope.distributed.types import DataFrame, JsonSerializableDataFrameModel


class SubjectGroupObservationsGDFSchema(JsonSerializableDataFrameModel):
    geometry: pa.typing.Series[Any] = pa.Field()   # see note in tasks/time_density re: geometry typing
    groupby_col: pa.typing.Series[object] = pa.Field()
    fixtime: pa.typing.Series[pd.DatetimeTZDtype] = pa.Field(dtype_kwargs={"unit": "ns", "tz": "UTC"})
    junk_status: pa.typing.Series[bool] = pa.Field()
    # TODO: can we be any more specific about the `extra__` field expectations?


@distributed
def get_subjectgroup_observations(
    # client
    server = Annotated[str, Field()],
    username = Annotated[str, Field()],
    password = Annotated[str, Field()],
    tcp_limit = Annotated[int, Field()],
    sub_page_size = Annotated[int, Field()],
    # get_subjectgroup_observations
    subject_group_name = Annotated[str, Field()],
    include_inactive = Annotated[bool, Field()],
    since = Annotated[str, Field()],
    until = Annotated[str, Field()],
) -> DataFrame[SubjectGroupObservationsGDFSchema]:
    from ecoscope.io import EarthRangerIO

    earthranger_io = EarthRangerIO(
        server=server,
        username=username,
        password=password,
        tcp_limit=tcp_limit,
        sub_page_size=sub_page_size,
    )
    return earthranger_io.get_subjectgroup_observations(
        subject_group_name=subject_group_name,
        include_subject_details=True,
        include_inactive=include_inactive,
        since=since,
        until=until,
    )
