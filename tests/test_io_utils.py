import json
import os
import zipfile
from http.client import HTTPMessage
from unittest.mock import MagicMock, Mock, patch

import fsspec
import pandas as pd
import pytest
from requests.exceptions import RetryError

import ecoscope
from ecoscope.io import utils

pytestmark = pytest.mark.io


def test_download_file_github_csv():
    ECOSCOPE_RAW = "https://raw.githubusercontent.com/wildlife-dynamics/ecoscope/master"
    output_dir = "tests/test_output"
    ecoscope.io.download_file(
        f"{ECOSCOPE_RAW}/tests/sample_data/vector/movebank_data.csv",
        os.path.join(output_dir, "download_data.csv"),
        overwrite_existing=True,
    )

    data = pd.read_csv(os.path.join(output_dir, "download_data.csv"))
    assert len(data) > 0


def test_download_file_gdrive_share_link():
    output_dir = "tests/test_output"
    ecoscope.io.download_file(
        "https://drive.google.com/file/d/1-AQ9_oacUCcAaiZ6SWU77hZWp1oArQw6/view?usp=drive_link",
        os.path.join(output_dir, "download_data.csv"),
        overwrite_existing=True,
    )

    data = pd.read_csv(os.path.join(output_dir, "download_data.csv"))
    assert len(data) > 0


def test_download_file_gdrive():
    output_dir = "tests/test_output"
    ecoscope.io.download_file(
        "https://drive.google.com/uc?export=download&id=1-AQ9_oacUCcAaiZ6SWU77hZWp1oArQw6",
        os.path.join(output_dir, "download_data.csv"),
        overwrite_existing=True,
    )

    data = pd.read_csv(os.path.join(output_dir, "download_data.csv"))
    assert len(data) > 0


def test_download_file_gdrive_zip():
    output_dir = "tests/test_output"
    ecoscope.io.download_file(
        "https://drive.google.com/uc?export=download&id=1YNQ6FBtlTAxmo8vmK59oTPBhAltI3kfK",
        output_dir,
        overwrite_existing=True,
        unzip=True,
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


def test_download_file_dropbox_share_link():
    output_dir = "tests/test_output"
    ecoscope.io.download_file(
        "https://www.dropbox.com/scl/fi/qaw3krcsnot69x94mdfxy/config.json?rlkey=zdmipl2la7rplgl218vc13end&dl=0",
        os.path.join(output_dir, "download_data.csv"),
        overwrite_existing=True,
    )

    data = pd.read_csv(os.path.join(output_dir, "download_data.csv"))
    assert len(data) > 0


@patch("urllib3.connectionpool.HTTPConnectionPool._get_conn")
def test_download_file_retry_on_error(mock):
    mock.return_value.getresponse.side_effect = [
        Mock(status=500, msg=HTTPMessage(), headers={}),
        Mock(status=504, msg=HTTPMessage(), headers={}),
        Mock(status=503, msg=HTTPMessage(), headers={}),
    ]

    url = "https://totallyreal.com"
    output_dir = "tests/test_output"

    with pytest.raises(RetryError):
        ecoscope.io.download_file(
            url,
            output_dir,
            overwrite_existing=True,
        )

    assert mock.call_count == 3


def _fake_response(body: bytes, content_disposition: str | None = None) -> MagicMock:
    r = MagicMock()
    headers: dict[str, str] = {}
    if content_disposition is not None:
        headers["content-disposition"] = content_disposition
    headers["content-length"] = str(len(body))
    r.headers = headers
    r.iter_content.return_value = [body]
    return r


def test_is_gdrive_url_matches() -> None:
    assert utils._is_gdrive_url("https://drive.google.com/file/d/abc123/view") is not None
    assert utils._is_gdrive_url("https://example.com/foo") is None


def test_is_dropbox_url_matches() -> None:
    assert utils._is_dropbox_url("https://www.dropbox.com/scl/fi/abc/name.csv?rlkey=xyz") is not None
    assert utils._is_dropbox_url("https://example.com/foo") is None


def test_transform_gdrive_url() -> None:
    out = utils._transform_gdrive_url("https://drive.google.com/file/d/FILEID/view?usp=drive_link")
    assert out == "https://drive.google.com/uc?export=download&id=FILEID"


def test_transform_dropbox_url() -> None:
    out = utils._transform_dropbox_url("https://www.dropbox.com/scl/fi/abc/name.csv?rlkey=xyz&dl=0")
    assert out.endswith("dl=1")


def test_download_file_writes_body(tmp_path) -> None:
    target = tmp_path / "out.bin"
    body = b"hello-world"

    with patch("ecoscope.io.utils.requests.Session") as MockSession:
        session = MockSession.return_value
        session.get.return_value = _fake_response(body)

        utils.download_file("https://example.com/x.bin", str(target))

    assert target.read_bytes() == body


def test_download_file_gdrive_url_transformed_before_request(tmp_path) -> None:
    target = tmp_path / "out.bin"

    with patch("ecoscope.io.utils.requests.Session") as MockSession:
        session = MockSession.return_value
        session.get.return_value = _fake_response(b"x")

        utils.download_file("https://drive.google.com/file/d/FILEID/view", str(target))

        called_url = session.get.call_args.args[0]
        assert called_url == "https://drive.google.com/uc?export=download&id=FILEID"


def test_download_file_dropbox_url_transformed_before_request(tmp_path) -> None:
    target = tmp_path / "out.bin"

    with patch("ecoscope.io.utils.requests.Session") as MockSession:
        session = MockSession.return_value
        session.get.return_value = _fake_response(b"x")

        utils.download_file(
            "https://www.dropbox.com/scl/fi/abc/name.csv?rlkey=xyz&dl=0",
            str(target),
        )

        called_url = session.get.call_args.args[0]
        assert called_url.endswith("dl=1")


def test_download_file_infers_filename_from_response_header(tmp_path) -> None:
    body = b"csv,body"

    with patch("ecoscope.io.utils.requests.Session") as MockSession:
        session = MockSession.return_value
        session.get.return_value = _fake_response(body, content_disposition='attachment; filename="inferred.csv"')

        utils.download_file("https://example.com/x", str(tmp_path))

    assert (tmp_path / "inferred.csv").read_bytes() == body


def test_download_file_raises_when_dir_and_no_filename(tmp_path) -> None:
    with patch("ecoscope.io.utils.requests.Session") as MockSession:
        session = MockSession.return_value
        session.get.return_value = _fake_response(b"x")

        with pytest.raises(ValueError, match="RFC 6266 filename"):
            utils.download_file("https://example.com/x", str(tmp_path))


def test_download_file_skips_existing_when_overwrite_false(tmp_path, capsys) -> None:
    target = tmp_path / "out.bin"
    target.write_bytes(b"original")

    with patch("ecoscope.io.utils.requests.Session") as MockSession:
        session = MockSession.return_value
        session.get.return_value = _fake_response(b"replacement")

        utils.download_file("https://example.com/x.bin", str(target), overwrite_existing=False)

        session.get.assert_not_called()

    assert target.read_bytes() == b"original"
    assert "Skipping" in capsys.readouterr().out


def test_download_file_unzip_extracts_contents(tmp_path) -> None:
    zip_target = tmp_path / "bundle.zip"
    inner_name = "inside.txt"
    inner_payload = b"hi-from-zip"

    buf_path = tmp_path / "src.zip"
    with zipfile.ZipFile(buf_path, "w") as zf:
        zf.writestr(inner_name, inner_payload)
    zip_bytes = buf_path.read_bytes()

    with patch("ecoscope.io.utils.requests.Session") as MockSession:
        session = MockSession.return_value
        session.get.return_value = _fake_response(zip_bytes)

        utils.download_file("https://example.com/bundle.zip", str(zip_target), unzip=True)

    assert (tmp_path / inner_name).read_bytes() == inner_payload
    assert os.path.exists(zip_target)
