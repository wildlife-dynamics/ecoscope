import geopandas as gpd  # type: ignore[import-untyped]
import pandas as pd
from ecoscope.platform.tasks.warning import mixed_subtype_warning
from shapely.geometry import Point
from wt_task import task
from wt_task.skip import SkipSentinel


def test_mixed_subtype_warning():
    same_subtype = gpd.GeoDataFrame(
        {
            "groupby_col": ["A", "B"],
            "fixtime": pd.to_datetime(["2024-01-01", "2025-01-01"], utc=True),
            "junk_status": [False, False],
            "extra__subject__name": ["Test", float("nan")],
            "extra__subject__subject_subtype": ["ranger", "ranger"],
            "geometry": [Point(0.0, 0.0), Point(100.0, 50.0)],
        }
    )
    mixed_subtype = same_subtype.copy()
    mixed_subtype["extra__subject__subject_subtype"] = ["ranger", "helicopter"]
    empty = pd.DataFrame()

    assert mixed_subtype_warning(same_subtype) is None
    assert mixed_subtype_warning(empty) is None
    assert task(mixed_subtype_warning).validate().call(SkipSentinel()) is None
    assert (
        mixed_subtype_warning(mixed_subtype)
        == "This workflow was run with mixed subtypes"
    )
