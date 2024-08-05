import pathlib
from dataclasses import dataclass
import papermill
from papermill.execute import PapermillExecutionError
import pytest


NB_DIR = pathlib.Path(__file__).parent.parent / "doc" / "source" / "notebooks"

KNOWN_ERRORS_REGEXES = {  # This is basically a GitHub ticket queue
    "EarthRanger_IO.ipynb": "Series found",
    "GEE_IO.ipynb": "Google Earth Engine API has not been used in project",
    "Relocations_and_Trajectories.ipynb": "not a zip file",
    "EcoGraph.ipynb": "not a zip file",
    "EcoMap.ipynb": "Google Earth Engine API has not been used in project",
    "EcoPlot.ipynb": "not a zip file",
    "Anomaly.ipynb": "Google Earth Engine API has not been used in project",
    "Calculation.ipynb": "Google Earth Engine API has not been used in project",
    "Chart.ipynb": "not a zip file",
}


@dataclass
class Notebook:
    path: pathlib.Path
    raises: bool = False
    raises_match: str = None


ALL_NOTEBOOKS = [
    Notebook(
        path=p,
        raises=False if not p.name in KNOWN_ERRORS_REGEXES else True,
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
    try:
        papermill.execute_notebook(str(notebook.path), "./output.ipynb", kernel_name="venv")
    except Exception as e:
        if notebook.raises:  # these are the ones we expect to raise errors
            pytest.xfail(f"Notebook {notebook.path} is known to fail with error {notebook.raises_match}")
        else:
            print(e.__traceback__)
            raise e  # this is not an expected error, so re-raise the captured error and make the test FAIL
    # with pytest.raises(Exception, match=notebook.raises_match):
            # First prove to ourselves that the notebook is still broken
            # then xfail when we know it is
        # Option 2: if papermill run in "capture errors" mode
        # error = papermill.execute_notebook(str(notebook), str(notebook.with_name("output.ipynb")))
        # assert notebook.expected_error_cell_number == error.cell_number
        # assert notebook_raises_match == error.text
        # pytest.xfail(f"Notebook {notebook.path} is known to fail with error {notebook.raises_match}")
    # papermill.execute_notebook(str(notebook), str(notebook.with_name("output.ipynb")))