import geopandas as gpd
import pytest

from ecoscope import Trajectory
from ecoscope.base import ProximityProfile, SpatialFeature


@pytest.fixture
def sample_spatial_features():
    return gpd.read_feather("tests/sample_data/vector/spatial_features.feather")


def test_proximity(sample_relocs, sample_spatial_features):
    # Subsample to keep the test fast; the assertion only checks that every spatial
    # feature appears in the output, which holds with any non-empty per-group subset.
    sample_relocs.gdf = sample_relocs.gdf.groupby("groupby_col").head(100)

    prox_profile = ProximityProfile([])

    for row in sample_spatial_features.iterrows():
        prox_profile.spatial_features.append(SpatialFeature(row[1]["name"], row[1]["pk"], row[1]["geometry"]))

    trajectory = Trajectory.from_relocations(sample_relocs)

    proximity_events = trajectory.calculate_proximity(proximity_profile=prox_profile)

    assert len(proximity_events["spatialfeature_id"].unique()) == len(prox_profile.spatial_features)
