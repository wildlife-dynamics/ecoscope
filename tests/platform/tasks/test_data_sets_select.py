from pathlib import Path

import geopandas as gpd  # type: ignore[import-untyped]
import pytest

import ecoscope.platform.tasks.io as tasks_io
from ecoscope.platform.schemas import RegionsGDFSchema
from ecoscope.platform.tasks.data_sets import gather_data_sets, read_data_set_table, select_spatial_features_group


@pytest.fixture
def all_groups_gdf():
    path = Path(tasks_io.__file__).parent / "get-spatial-features-groups.example-return.parquet"
    return gpd.read_parquet(path)


def test_select_returns_only_the_named_group(all_groups_gdf):
    out = select_spatial_features_group(groups=all_groups_gdf, spatial_features_group_name="Sectors")
    assert len(out) == 2 and {m["display_name"] for m in out["metadata"]} == {"Sectors"}
    RegionsGDFSchema.validate(out)


def test_select_unknown_name_raises_with_available_names(all_groups_gdf):
    with pytest.raises(ValueError, match=r"'Zones' not found.*available: Blocks, Sectors"):
        select_spatial_features_group(groups=all_groups_gdf, spatial_features_group_name="Zones")


def test_select_on_unbound_groups_explains_the_binding(all_groups_gdf):
    with pytest.raises(ValueError, match="the bound data set has none"):
        select_spatial_features_group(groups=all_groups_gdf.iloc[0:0], spatial_features_group_name="Sectors")


def test_select_with_no_name_returns_empty_frame(all_groups_gdf):
    assert len(select_spatial_features_group(groups=all_groups_gdf, spatial_features_group_name="")) == 0
    assert len(select_spatial_features_group(groups=all_groups_gdf.iloc[0:0], spatial_features_group_name="")) == 0


def test_select_works_on_a_data_set_round_trip(tmp_path, all_groups_gdf):
    gather_data_sets(data_set="spatial", tables={"spatial_feature_groups": all_groups_gdf}, root_path=str(tmp_path))
    roots = f'{{"spatial_feature_groups": "{tmp_path}"}}'
    groups = read_data_set_table(data_set_roots=roots, table_type="spatial_feature_groups")
    out = select_spatial_features_group(groups=groups, spatial_features_group_name="Sectors")
    assert len(out) == 2
    RegionsGDFSchema.validate(out)
