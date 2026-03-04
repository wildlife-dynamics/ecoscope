import pytest
from ecoscope.platform.tasks.io import download_roi

pytestmark = pytest.mark.io


def test_download_roi():
    result = download_roi(url="https://www.dropbox.com/s/nvdmidz1o2duyl3/AOIs.gpkg?dl=1")
    assert len(result) > 0
    assert "name" == result.index.name
    assert "geometry" in result
