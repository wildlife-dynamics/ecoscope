from pathlib import Path
from unittest.mock import MagicMock

import geopandas as gpd  # type: ignore[import-untyped]
import pytest

import ecoscope.platform
from ecoscope.platform.connections import EarthRangerBackends
from ecoscope.platform.schemas import RegionsGDFSchema
from ecoscope.platform.tasks.io import get_spatial_features_groups

FIXTURE_DIR = Path(ecoscope.platform.tasks.io.__file__).parent


def _group_payload(group_id: str, name: str, features: gpd.GeoDataFrame) -> dict:
    return {
        "id": group_id,
        "name": name,
        "description": "",
        "url": f"https://er.example/api/v1.0/spatialfeaturegroup/{group_id}",
        "feature_count": len(features),
        "created_at": "2024-01-01T00:00:00Z",
        "updated_at": "2024-01-01T00:00:00Z",
        "features": features,
    }


@pytest.fixture
def spatial_features_group_payloads():
    fixture = gpd.read_parquet(FIXTURE_DIR / "get-spatial-features-group.example-return.parquet").reset_index(drop=True)
    return [
        _group_payload("group-blocks", "Blocks", fixture.iloc[:2].copy()),
        _group_payload("group-sectors", "Sectors", fixture.iloc[1:3].copy()),
    ]


def test_get_spatial_features_groups_concatenates_every_group(
    spatial_features_group_payloads,
):
    api = MagicMock()
    api.get_spatial_features_groups.return_value = spatial_features_group_payloads
    gdf = get_spatial_features_groups(client=EarthRangerBackends(api=api, warehouse=None))
    assert len(gdf) == 4
    assert sorted({m["display_name"] for m in gdf["metadata"]}) == [
        "Blocks",
        "Sectors",
    ]
    assert {m["id"] for m in gdf["metadata"]} == {"group-blocks", "group-sectors"}
    RegionsGDFSchema.validate(gdf)


def test_get_spatial_features_groups_site_without_groups():
    api = MagicMock()
    api.get_spatial_features_groups.return_value = []
    gdf = get_spatial_features_groups(client=EarthRangerBackends(api=api, warehouse=None))
    assert gdf.empty


def test_get_spatial_features_groups_example_return_selects_by_name():
    gdf = gpd.read_parquet(FIXTURE_DIR / "get-spatial-features-groups.example-return.parquet")
    RegionsGDFSchema.validate(gdf)
    assert len(gdf) == 4
    for name in ("Blocks", "Sectors"):
        selected = gdf[gdf.metadata.apply(lambda m: m["display_name"]) == name]
        assert len(selected) == 2
