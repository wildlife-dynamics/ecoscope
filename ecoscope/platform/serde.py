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


def _reroot(path: str, root_path: str) -> str:
    """Return the read path for a `path` relative to `root_path`; leave anything absolute as-is.

    Persisting returns paths relative to the results root, so that a value -- and
    the step-cache digest of everything built from it -- does not change with a
    per-run root. This is the inverse, for whatever finally hands paths to a
    reader. "Absolute" includes any URL scheme, which on Windows also catches a
    drive letter.

    Examples:

    ```python
    >>> _reroot("file.txt", "gs://bucket/path/to/dir")
    'https://storage.googleapis.com/bucket/path/to/dir/file.txt'
    >>> _reroot("https://example.com/file.txt", "gs://bucket/path/to/dir")
    'https://example.com/file.txt'
    >>> _reroot("/tmp/elsewhere/file.txt", "gs://bucket/path/to/dir")
    '/tmp/elsewhere/file.txt'

    ```

    """
    if urlparse(path).scheme or Path(path).is_absolute():
        return path
    return _get_path(root_path, path)[1]


def _read_path_if_file_exists(root_path: str, filename: str) -> str | None:
    """Return `filename` if it already exists under `root_path` and is a usable file.

    Returns None when the target is absent, is not a regular file, or is an empty local file.

    Note: as with `_get_path`, resolving a local `root_path` creates the directory
    as a side effect.
    """
    write_path, _ = _get_path(root_path, filename)
    if not write_path.is_file():
        return None
    if isinstance(write_path, Path) and write_path.stat().st_size == 0:
        return None
    return filename


def _persist_text(text: str, root_path: str, filename: str) -> str:
    """Write `text` to `filename` under `root_path`, and return `filename` (see `_reroot`)."""
    write_path, _ = _get_path(root_path, filename)

    try:
        write_path.write_text(text, encoding="utf-8")
    except Exception as e:
        raise ValueError(f"Failed to write text to {write_path}") from e

    return filename


def _persist_bytes(data: bytes, root_path: str, filename: str) -> str:
    """Write `data` to `filename` under `root_path`, and return `filename` (see `_reroot`)."""
    write_path, _ = _get_path(root_path, filename)

    try:
        write_path.write_bytes(data)
    except Exception as e:
        raise ValueError(f"Failed to write bytes to {write_path}") from e

    return filename
