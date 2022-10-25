import os

import geopandas as gpd
import geopandas.testing
import numpy as np
import pandas as pd
import pytest

import ecoscope

def test_recurse(movbank_relocations):
    # apply relocation coordinate filter to movbank data
    pnts_filter = ecoscope.base.RelocsCoordinateFilter(
        min_x=-5,
        max_x=1,
        min_y=12,
        max_y=18,
        filter_point_coords=[[180, 90], [0, 0]],
    )
    movbank_relocations.apply_reloc_filter(pnts_filter, inplace=True)
    movbank_relocations.remove_filtered(inplace=True)

    revisits = movbank_relocations.copy()
    revisits["revisits"] = ecoscope.analysis.get_recursions(movbank_relocations, resolution=400)

    expected_revisits = gpd.read_feather("tests/test_output/salif_revisits.feather")

    gpd.testing.assert_geodataframe_equal(revisits, expected_revisits, check_less_precise=True)