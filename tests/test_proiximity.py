import pytest
from ecoscope.analysis.proximity import SpatialFeature, Proximity, ProximityProfile
from ecoscope.base import Trajectory


@pytest.mark.skipif(not pytest.earthranger, reason="No connection to EarthRanger")
def test_proximity(er_io):
    er_features = er_io.get_spatial_features_group(spatial_features_group_id="15698426-7e0f-41df-9bc3-495d87e2e097")

    prox_profile = ProximityProfile([])

    for row in er_features.iterrows():
        prox_profile.spatial_features.append(SpatialFeature(row[1]["name"], row[1]["pk"], row[1]["geometry"]))

    relocations = er_io.get_subjectgroup_observations(subject_group_name=er_io.GROUP_NAME)
    trajectory = Trajectory.from_relocations(relocations)

    proximity_events = Proximity.calculate_proximity(proximity_profile=prox_profile, trajectory=trajectory)

    assert len(proximity_events["spatialfeature_id"].unique()) == len(prox_profile.spatial_features)
