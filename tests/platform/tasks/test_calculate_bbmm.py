from importlib.resources import files

import geopandas as gpd  # type: ignore[import-untyped]
import pytest
from pydantic import TypeAdapter

from ecoscope.platform.tasks.analysis import calculate_brownian_bridge_range
from ecoscope.platform.tasks.analysis._time_density import TimeDensityReturnGDF


@pytest.fixture(scope="module")
def trajectory_gdf():
    example_input_df_path = (
        files("ecoscope.platform.tasks.preprocessing") / "relocations-to-trajectory.example-return.parquet"
    )
    gdf = gpd.read_parquet(example_input_df_path)
    # Subsampled to keep this test's runtime low; BBMM's per-segment grid
    # accumulation is considerably slower than ETD's on the same fixture.
    return gdf.iloc[::10].copy()


def test_calculate_brownian_bridge_range_default_percentiles(trajectory_gdf):
    result = calculate_brownian_bridge_range(trajectory_gdf, crs="ESRI:102022")

    assert list(result.columns) == ["percentile", "geometry", "area_sqkm"]
    assert len(result) == 7  # default percentiles: 50/60/70/80/90/95/99.999
    ta = TypeAdapter(TimeDensityReturnGDF)
    ta.validate_python(result)


def test_calculate_brownian_bridge_range_area_increases_with_percentile(trajectory_gdf):
    result = calculate_brownian_bridge_range(trajectory_gdf, crs="ESRI:102022", percentiles=[50.0, 90.0])

    area_by_percentile = result.set_index("percentile")["area_sqkm"]
    assert area_by_percentile[90.0] > area_by_percentile[50.0]


def test_calculate_brownian_bridge_range_raises_on_empty_percentiles(trajectory_gdf):
    with pytest.raises(ValueError, match="cannot be empty"):
        calculate_brownian_bridge_range(trajectory_gdf, crs="ESRI:102022", percentiles=[])
