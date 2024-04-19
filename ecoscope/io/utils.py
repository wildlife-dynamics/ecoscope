import email
import os
import re
import typing
import zipfile

import pandas as pd
import requests
from tqdm.auto import tqdm


def to_hex(val, default="#ff0000"):
    if val and not pd.isnull(val):
        return "#{:02X}{:02X}{:02X}".format(*[int(i) for i in val.split(",")])
    return default


def pack_columns(dataframe: pd.DataFrame, columns: typing.List):
    """This method would add all extra columns to single column"""
    metadata_cols = list(set(dataframe.columns).difference(set(columns)))

    # To prevent additional column from being dropped, name the column metadata (rename it back).
    if metadata_cols:
        dataframe["metadata"] = dataframe[metadata_cols].to_dict(orient="records")
        dataframe.drop(metadata_cols, inplace=True, axis=1)
        dataframe.rename(columns={"metadata": "additional"}, inplace=True)
    return dataframe


def download_file(url, path, overwrite_existing=False, chunk_size=1024, unzip=True, **request_kwargs):
    """
    Download a file from a URL to a local path. If the path is a directory, the filename will be inferred from
    the response header
    """

    if __is_gdrive_url(url):
        url = __transform_gdrive_url(url)
    elif __is_dropbox_url(url):
        url = __transform_dropbox_url(url)

    r = requests.get(url, stream=True, **request_kwargs)

    if os.path.isdir(path):
        m = email.message.Message()
        m["content-type"] = r.headers["content-disposition"]
        filename = m.get_param("filename")
        if filename is None:
            raise ValueError("URL has no RFC 6266 filename.")
        path = os.path.join(path, filename)

    if os.path.exists(path) and not overwrite_existing:
        print(f"{path} exists. Skipping...")
        return

    with open(path, "wb") as f:
        with tqdm.wrapattr(f, "write", total=int(r.headers["Content-Length"])) as fout:
            for chunk in r.iter_content(chunk_size=chunk_size):
                fout.write(chunk)

    # Check if the file is a zip file
    if zipfile.is_zipfile(path) and unzip:
        # Unzip the file
        with zipfile.ZipFile(path, "r") as zip_ref:
            zip_ref.extractall(os.path.dirname(path))


def __is_gdrive_url(url):
    pattern = r"https://drive\.google\.com/file/d/(.*?)"
    return re.match(pattern, url)


def __is_dropbox_url(url):
    pattern = r"https://www\.dropbox\.com/scl/fi/(.*?)/(.*?)\?rlkey=(.*?)"
    return re.match(pattern, url)


def __transform_gdrive_url(url):
    file_id = url.split("/d/")[1].split("/")[0]
    return "https://drive.google.com/uc?export=download&id=" + file_id


def __transform_dropbox_url(url):
    return url[:-1] + "1"
