import os
import typing

import fsspec
import pandas as pd

# import zipfile


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


def download_file(url, path, overwrite_existing=False, unzip=True):
    """
    Download a file from a URL to a local path. If the path is a directory, the filename will be inferred from
    the response header
    """
    fs = fsspec.filesystem("http")
    # fs = fsspec.filesystem("dropbox")
    # todo: infer filename
    # fs, fs_token, path = fsspec.core.url_to_fs(url)

    # If path is a directory, infer filename from URL
    if os.path.isdir(path):
        filename = url.split("/")[-1]
        path = os.path.join(path, filename)

    if os.path.exists(path) and not overwrite_existing:
        print(f"{path} exists. Skipping...")
        return

    with fs.open(url, "rb") as src:
        with open(path, "wb") as dst:
            dst.write(src.read())

    # Check if the file is a zip file
    # if zipfile.is_zipfile(path) and unzip:
    #     # Unzip the file
    #     with zipfile.ZipFile(path, 'r') as zip_ref:
    #         zip_ref.extractall(os.path.dirname(path))

    # r = requests.get(url, stream=True, **request_kwargs)

    # if os.path.isdir(path):
    #     m = email.message.Message()
    #     m["content-type"] = r.headers["content-disposition"]
    #     filename = m.get_param("filename")
    #     if filename is None:
    #         raise ValueError("URL has no RFC 6266 filename.")
    #     path = os.path.join(path, filename)

    # if os.path.exists(path) and not overwrite_existing:
    #     print(f"{path} exists. Skipping...")
    #     return

    # with open(path, "wb") as f:
    #     with tqdm.wrapattr(f, "write", total=int(r.headers["Content-Length"])) as fout:
    #         for chunk in r.iter_content(chunk_size=chunk_size):
    #             fout.write(chunk)
