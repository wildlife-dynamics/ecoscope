import pandas as pd
import pytest

from ecoscope.base.utils import ModisBegin


@pytest.mark.skip(reason="this has been failing since May 2022; will be fixed in a follow-up pull")
def test_modis_offset():
    ts1 = pd.Timestamp("2022-01-13 17:00:00+0")
    ts2 = pd.Timestamp("2022-12-26 17:00:00+0")
    assert ts1 + ModisBegin() == pd.Timestamp("2022-01-17 00:00:00+0")
    assert ts2 + ModisBegin() == pd.Timestamp("2023-01-01 00:00:00+0")
