import mimetypes
import os
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import quote, urlparse


def _gs_url_to_https_url(gs_url: str):
    assert gs_url.startswith("gs://")
    https_url = gs_url.replace("gs://", "https://storage.googleapis.com/")
    parts = https_url.split("/")
    encoded_parts = [quote(part, safe="") for part in parts[4:]]
    return "/".join(parts[:4] + encoded_parts)


def _my_content_type(path: str) -> tuple[str | None, str | None]:
    # xref https://cloudpathlib.drivendata.org/stable/other_client_settings/
    return mimetypes.guess_type(path)


def _get_path(root_path: str, filename: str):
    """Given a root path and a filename, return a 2-tuple of write and read paths.

    Examples:

    ```python
    >>> _get_path("gs://bucket/path/to/dir", "file.txt")
    (GSPath('gs://bucket/path/to/dir/file.txt'), 'https://storage.googleapis.com/bucket/path/to/dir/file.txt')
    >>> _get_path("file:///tmp/ecoscope-workflows/test/dir", "file.txt")
    (PosixPath('/tmp/ecoscope-workflows/test/dir/file.txt'), '/tmp/ecoscope-workflows/test/dir/file.txt')

    ```

    """
    if TYPE_CHECKING:
        from cloudpathlib.gs.gspath import GSPath

        write_path: Path | "GSPath"

    parsed_url = urlparse(root_path)
    match parsed_url.scheme:
        case "file" | "":
            # Handle Windows file URLs properly
            if os.name == "nt":
                local_path = Path(root_path.lstrip("file://"))
            else:
                # Standard file URL or local path
                local_path = Path(parsed_url.path)
            if not local_path.exists():
                local_path.mkdir(parents=True, exist_ok=True)
            write_path = local_path / filename
            read_path = write_path.absolute().as_posix()

        case "gs":
            from cloudpathlib.gs.gsclient import GSClient
            from cloudpathlib.gs.gspath import GSPath

            client = GSClient(content_type_method=_my_content_type)
            client.set_as_default_client()

            write_path = GSPath(root_path) / filename
            read_path = _gs_url_to_https_url(write_path.as_uri())
        case _:
            raise ValueError(f"Unsupported scheme for: {root_path}")
    return write_path, read_path


def _read_path_if_file_exists(root_path: str, filename: str) -> str | None:
    """The read path for `filename` under `root_path`, if it is already a usable file.

    Lets a caller skip a redundant write and still hand back the same path
    `_persist_text` / `_persist_bytes` would have returned. Only safe for callers
    whose `filename` fully determines the content (content-addressed / hash-derived
    names), since the existing bytes stand in for the ones not written.

    None when the target is absent, is not a regular file, or is an empty local
    file (the usual artifact of a crash mid-write; local writes are not atomic,
    GCS uploads are).

    `is_file()` rather than `exists()`: on GS a bare prefix match reports as
    existing, and locally a directory would too -- which would silently skip a
    write that should raise.

    Note: as with `_get_path`, resolving a local `root_path` creates the directory
    as a side effect.
    """
    write_path, read_path = _get_path(root_path, filename)
    if not write_path.is_file():
        return None
    if isinstance(write_path, Path) and write_path.stat().st_size == 0:
        return None
    return read_path


def _persist_text(text: str, root_path: str, filename: str) -> str:
    write_path, read_path = _get_path(root_path, filename)

    try:
        write_path.write_text(text, encoding="utf-8")
    except Exception as e:
        raise ValueError(f"Failed to write text to {write_path}") from e

    return read_path


def _persist_bytes(data: bytes, root_path: str, filename: str) -> str:
    write_path, read_path = _get_path(root_path, filename)

    try:
        write_path.write_bytes(data)
    except Exception as e:
        raise ValueError(f"Failed to write bytes to {write_path}") from e

    return read_path
