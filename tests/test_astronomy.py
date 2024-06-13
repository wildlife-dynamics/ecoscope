import pyproj
import pandas as pd
from ecoscope.base import Trajectory
from ecoscope.analysis import astronomy


def test_to_EarthLocation(movebank_relocations):
    geometry = movebank_relocations["geometry"]
    test_point = geometry.iloc[0]

    transformed = astronomy.to_EarthLocation(geometry)

    assert len(geometry) == len(transformed)

    transform = pyproj.Transformer.from_proj(
        proj_from="+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs",
        proj_to="+proj=geocent +ellps=WGS84 +datum=WGS84 +units=m +no_defs",
    )

    # Check the projected values in the returned EarthLocation are what we expect
    test_val = transform.transform(xx=test_point.x, yy=test_point.y, zz=0)
    assert test_val[0] == transformed[0].x.value
    assert test_val[1] == transformed[0].y.value
    assert test_val[2] == transformed[0].z.value


def test_is_night(movebank_relocations):
    subset = movebank_relocations.iloc[12:15].copy()

    subset["is_night"] = astronomy.is_night(subset.geometry, subset.fixtime)

    assert subset["is_night"].values.tolist() == [True, True, False]


def test_daynight_ratio(movebank_relocations):
    trajectory = Trajectory.from_relocations(movebank_relocations)
    expected = pd.Series(
        [
            2.212816,
            0.656435,
        ],
        index=pd.Index(["Habiba", "Salif Keita"], name="groupby_col"),
    )
    pd.testing.assert_series_equal(
        trajectory.groupby("groupby_col")[trajectory.columns].apply(astronomy.get_daynight_ratio, include_groups=False),
        expected,
    )
