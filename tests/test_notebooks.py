import os
import pathlib
from dataclasses import dataclass
import papermill
from papermill.execute import PapermillExecutionError
import pytest


NB_DIR = pathlib.Path(__file__).parent.parent / "doc" / "source" / "notebooks"

KNOWN_ERRORS_REGEXES = {  # This is basically a GitHub ticket queue
    "EarthRanger_IO.ipynb": "Series found",
    "GEE_IO.ipynb": "frontend does not support input requests", # This error only happens in the ci pipeline
    "EcoMap.ipynb": "frontend does not support input requests", # This error only happens in the ci pipeline
    "Remote Sensing Time Series Anomaly.ipynb": "frontend does not support input requests", # This error only happens in the ci pipeline
    "Relocations_and_Trajectories.ipynb": "not a zip file",
    "EcoGraph.ipynb": "not a zip file",
    "EcoPlot.ipynb": "not a zip file",
    "Landscape Grid.ipynb": "not a zip file",
    "Seasonal Calculation.ipynb": "buffer source array is read-only",
    "Tracking Data Gantt Chart.ipynb": "not a zip file",
}
class UnexecptedNotebookExecutionError(Exception):
    pass

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
@pytest.mark.skipif(os.environ.get("SKIP_NB_TESTS"), reason="Nb tests should not run for this pipeline")
def test_notebooks(notebook: Notebook):
    try:
        papermill.execute_notebook(str(notebook.path), "./output.ipynb", kernel_name="venv")
    except Exception as e:
        if notebook.raises:  # these are the ones we expect to raise errors
            pytest.xfail(f"Notebook {notebook.path} is known to fail with error {notebook.raises_match}")
        else:
            raise UnexecptedNotebookExecutionError(
                f"{notebook.path.name=} not in {list(KNOWN_ERRORS_REGEXES)= } but execution errored. "
                "This notebook is unexpectedly broken."
            ) from e