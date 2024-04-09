import json
import os

import fsspec
import pandas as pd

import ecoscope


def test_download_file_github_csv():
    ECOSCOPE_RAW = "https://raw.githubusercontent.com/wildlife-dynamics/ecoscope/master"
    output_dir = "tests/test_output"
    ecoscope.io.download_file(
        f"{ECOSCOPE_RAW}/tests/sample_data/vector/movbank_data.csv",
        os.path.join(output_dir, "movbank_data.csv"),
        overwrite_existing=True,
    )

    data = pd.read_csv(os.path.join(output_dir, "movbank_data.csv"))
    assert len(data) > 0


def test_download_file_dropbox_json():
    URL = "https://www.dropbox.com/scl/fi/qaw3krcsnot69x94mdfxy/config.json?rlkey=zdmipl2la7rplgl218vc13end&dl=1"
    output_path = "tests/test_output/config.json"
    ecoscope.io.download_file(URL, output_path)

    with fsspec.open(
        output_path,
        mode="rt",
    ) as file:
        config = json.loads(file.read())
        assert config is not None
