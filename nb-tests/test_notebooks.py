import os
import re
import pathlib
from dataclasses import dataclass
import ee
import papermill
import pytest

try:
    EE_ACCOUNT = os.getenv("EE_ACCOUNT")
    EE_PRIVATE_KEY_DATA = os.getenv("EE_PRIVATE_KEY_DATA")
    if EE_ACCOUNT and EE_PRIVATE_KEY_DATA:
        ee.Initialize(credentials=ee.ServiceAccountCredentials(EE_ACCOUNT, key_data=EE_PRIVATE_KEY_DATA))
except Exception:
    raise ValueError("Earth Engine can not be initialized. Failing notebook tests")

NB_DIR = pathlib.Path(__file__).parent.parent / "doc" / "source" / "notebooks"

KNOWN_ERRORS_REGEXES = {}  # This is basically a GitHub ticket queue


@dataclass
class Notebook:
    path: pathlib.Path
    raises: bool = False
    raises_match: str = None


ALL_NOTEBOOKS = [
    Notebook(
        path=p,
        raises=False if p.name not in KNOWN_ERRORS_REGEXES else True,
        raises_match=KNOWN_ERRORS_REGEXES.get(p.name),
    )
    for p in NB_DIR.rglob("*.ipynb")
]


@pytest.mark.parametrize(
    "notebook",
    ALL_NOTEBOOKS,
    ids=[nb.path.name for nb in ALL_NOTEBOOKS],
)
def test_notebooks(notebook: Notebook):
    if notebook.raises:  # these are the ones we expect to raise errors
        with pytest.raises(Exception, match=re.escape(notebook.raises_match)):
            papermill.execute_notebook(str(notebook.path), "./output.ipynb", kernel_name="venv")
        pytest.xfail(f"Notebook {notebook.path} is known to fail with error {notebook.raises_match}")
    papermill.execute_notebook(str(notebook.path), "./output.ipynb", kernel_name="venv")
