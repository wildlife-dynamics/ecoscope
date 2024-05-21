import pandas as pd
import pytest

from ecoscope.analysis import immobility


def test_immobility(movebank_relocations):
    movebank_relocations = movebank_relocations.loc[movebank_relocations.groupby_col == "Salif Keita"][:100]
    immobility_profile = immobility.ImmobilityProfile(
        threshold_time=130, threshold_probability=0.5, threshold_radius=1000
    )
    result = immobility.Immobility.calculate_immobility(
        immobility_profile=immobility_profile, relocs=movebank_relocations
    )
    expected_result = {
        "probability_value": 0.83,
        "cluster_radius": 1913.5975702951566,
        "cluster_fix_count": 83,
        "total_fix_count": 100,
        "immobility_time": pd.Timedelta("4 days 03:00:00"),
    }

    assert result == pytest.approx(expected_result)
