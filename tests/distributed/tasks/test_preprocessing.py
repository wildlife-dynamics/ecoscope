from pathlib import Path

import geopandas as gpd
import pandas as pd
import pytest

from ecoscope.distributed.tasks.preprocessing import process_relocations


@pytest.fixture
def observations_parquet_path() -> str:
    return Path(__file__).parent.parent / "data" / "subject-group.parquet"


def test_process_relocations(observations_parquet_path: str, tmp_path):
    observations = gpd.read_parquet(observations_parquet_path)
    kws = dict(
        filter_point_coords=[[180, 90], [0, 0]],
        relocs_columns=["groupby_col", "fixtime", "junk_status", "geometry"],
    )
    in_memory = process_relocations(observations, **kws)

    # compare to `distributed` calling style
    def serialize_result(gdf: gpd.GeoDataFrame) -> str:
        path: Path = tmp_path / "result.parquet"
        gdf.to_parquet(path)
        return path.as_posix()

    distributed_kws = dict(
        arg_prevalidators={"observations": lambda path: gpd.read_parquet(path)},
        return_postvalidator=serialize_result,
        validate=True,
    )
    result_path = process_relocations.replace(**distributed_kws)(observations_parquet_path, **kws)
    distributed_result = gpd.read_parquet(result_path)

    pd.testing.assert_frame_equal(in_memory, distributed_result)

    # we've cached this result for downstream tests, to make sure the cache is not stale
    cached = gpd.read_parquet(Path(__file__).parent.parent / "data" / "relocations.parquet")
    pd.testing.assert_frame_equal(in_memory, cached)
