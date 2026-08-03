from importlib.resources import files

import geopandas as gpd  # type: ignore[import-untyped]
import pytest
from pydantic import TypeAdapter

from ecoscope.platform.tasks.analysis import calculate_minimum_convex_polygon
from ecoscope.platform.tasks.analysis._time_density import TimeDensityReturnGDF


@pytest.fixture
def relocations_gdf():
    example_input_df_path = (
        files("ecoscope.platform.tasks.preprocessing") / "process-relocations.example-return.parquet"
    )
    return gpd.read_parquet(example_input_df_path)


def test_calculate_minimum_convex_polygon_default_percentiles(relocations_gdf):
    result = calculate_minimum_convex_polygon(relocations_gdf)

    assert list(result.columns) == ["percentile", "geometry", "area_sqkm"]
    assert len(result) == 7  # default percentiles: 50/60/70/80/90/95/99.999
    ta = TypeAdapter(TimeDensityReturnGDF)
    ta.validate_python(result)


def test_calculate_minimum_convex_polygon_area_increases_with_percentile(relocations_gdf):
    result = calculate_minimum_convex_polygon(relocations_gdf, percentiles=[50.0, 90.0])

    area_by_percentile = result.set_index("percentile")["area_sqkm"]
    assert area_by_percentile[90.0] > area_by_percentile[50.0]


def test_calculate_minimum_convex_polygon_dedupes_and_sorts_percentiles(relocations_gdf):
    result = calculate_minimum_convex_polygon(relocations_gdf, percentiles=[90.0, 50.0, 90.0])

    assert list(result["percentile"]) == [90.0, 50.0]


def test_calculate_minimum_convex_polygon_raises_on_empty_percentiles(relocations_gdf):
    with pytest.raises(ValueError, match="cannot be empty"):
        calculate_minimum_convex_polygon(relocations_gdf, percentiles=[])
