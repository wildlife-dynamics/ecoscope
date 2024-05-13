import os
from datetime import datetime
from pathlib import Path

import geopandas as gpd
import pandas as pd

from ecoscope.distributed.tasks.io import get_subjectgroup_observations


def test_subjectgroup_observations(tmp_path):
    kws = dict(
        server=os.environ["ER_SERVER"],
        username=os.environ["ER_USERNAME"],
        password=os.environ["ER_PASSWORD"],
        tcp_limit=5,
        sub_page_size=4000,
        # get_subjectgroup_observations
        subject_group_name="Elephants",
        include_inactive=True,
        since=datetime.strptime("2011-01-01", "%Y-%m-%d"),
        until=datetime.strptime("2023-01-01", "%Y-%m-%d"),
    )
    in_memory = get_subjectgroup_observations(**kws)

    assert all([column in in_memory for column in ["geometry", "groupby_col", "fixtime", "junk_status"]])

    # compare to `distributed` calling style; in this mode, we return *a path*, not a GeoDataFrame
    def serialize_result(gdf: gpd.GeoDataFrame) -> str:
        path: Path = tmp_path / "result.parquet"
        gdf.to_parquet(path)
        return path.as_posix()

    distributed_kws = dict(return_postvalidator=serialize_result, validate=True)
    result_path = get_subjectgroup_observations.replace(**distributed_kws)(**kws)
    distributed_result = gpd.read_parquet(result_path)

    pd.testing.assert_frame_equal(in_memory, distributed_result)

    # we've cached this result to speed up downstream tests, to make sure the cache is not stale
    cached = gpd.read_parquet(Path(__file__).parent.parent / "data" / "subject-group.parquet")
    pd.testing.assert_frame_equal(in_memory, cached)
