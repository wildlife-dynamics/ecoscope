import geopandas as gpd
import geopandas.testing
import pandas as pd
from shapely.geometry import MultiLineString, Polygon

import ecoscope
from ecoscope.analysis import geofence


def test_geofence_crossing():
    obs = gpd.read_file("tests/sample_data/vector/observations.geojson")
    obs["recorded_at"] = pd.to_datetime(obs["recorded_at"], utc=True)
    relocations = ecoscope.base.Relocations.from_gdf(obs, groupby_col="source_id", time_col="recorded_at")
    trajectory = ecoscope.base.Trajectory.from_relocations(relocations)

    region = Polygon(
        [
            [18.45703125, 4.609278084409835],
            [18.6767578125, 4.521666342614804],
            [18.6328125, 4.653079918274051],
            [17.4462890625, -1.1425024037061522],
            [36.6943359375, -4.653079918274038],
            [41.0888671875, -1.9332268264771106],
            [41.66015625, 4.039617826768437],
            [18.45703125, 4.609278084409835],
        ]
    )

    equator_fence = MultiLineString([((0, 0), (90, 0)), ((180, 0), (270, 0))])

    geofences = [
        geofence.GeoFence(
            geometry=equator_fence,
            unique_id="f81bff6a-9906-4a22-88cc-aa7ab958d876",
            fence_name="equator_fence",
            warn_level="warning",
        )
    ]

    regions = [
        geofence.Region(
            region_name="East & Central Africa",
            geometry=region,
            unique_id="1c24d330-c5df-4fd6-aa78-1e07f7b14edd",
        )
    ]

    gf_profile = geofence.GeoCrossingProfile(geofences=geofences, regions=regions)
    df = geofence.GeoFenceCrossing.analyse(geocrossing_profile=gf_profile, trajectory=trajectory)
    geofence_crossing_point = gpd.read_feather("tests/test_output/geofence_crossing_point.feather")
    gpd.testing.assert_geodataframe_equal(df, geofence_crossing_point)
    assert df.crs == geofence_crossing_point.crs
