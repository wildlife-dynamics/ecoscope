import email
import os
import re
import zipfile

import requests
from requests.adapters import HTTPAdapter
from tqdm.auto import tqdm
from urllib3.util import Retry


def download_file(
    url: str,
    path: str,
    retries: int = 2,
    overwrite_existing: bool = False,
    chunk_size: int = 1024,
    unzip: bool = False,
    **request_kwargs,
) -> None:
    """
    Download a file from a URL to a local path. If the path is a directory, the filename will be inferred from
    the response header
    """

    s = requests.Session()
    max_retries = Retry(total=retries, backoff_factor=0.1, status_forcelist=[500, 502, 503, 504])
    s.mount("https://", HTTPAdapter(max_retries=max_retries))

    if __is_gdrive_url(url):
        url = __transform_gdrive_url(url)
    elif __is_dropbox_url(url):
        url = __transform_dropbox_url(url)

    r = s.get(url, stream=True, **request_kwargs)

    if os.path.isdir(path):
        m = email.message.Message()
        m["content-type"] = r.headers.get("content-disposition", m.get_default_type())
        filename = m.get_param("filename")
        if filename is None:
            raise ValueError("URL has no RFC 6266 filename.")
        path = os.path.join(path, filename)  # type: ignore[arg-type]

    if os.path.exists(path) and not overwrite_existing:
        print(f"{path} exists. Skipping...")
        return

    with open(path, "wb") as f:
        content_length = r.headers.get("content-length")
        with tqdm.wrapattr(f, "write", total=int(content_length)) if content_length else f as fout:
            for chunk in r.iter_content(chunk_size=chunk_size):
                fout.write(chunk)

    # Check if the file is a zip file
    if zipfile.is_zipfile(path) and unzip:
        # Unzip the file
        with zipfile.ZipFile(path, "r") as zip_ref:
            zip_ref.extractall(os.path.dirname(path))


def __is_gdrive_url(url: str) -> re.Match | None:
    pattern = r"https://drive\.google\.com/file/d/(.*?)"
    return re.match(pattern, url)


def __is_dropbox_url(url: str) -> re.Match | None:
    pattern = r"https://www\.dropbox\.com/scl/fi/(.*?)/(.*?)\?rlkey=(.*?)"
    return re.match(pattern, url)


def __transform_gdrive_url(url: str) -> str:
    file_id = url.split("/d/")[1].split("/")[0]
    return "https://drive.google.com/uc?export=download&id=" + file_id


def __transform_dropbox_url(url: str) -> str:
    return url[:-1] + "1"
