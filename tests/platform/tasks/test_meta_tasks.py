from ecoscope.platform.tasks.config import (
    get_opacity_from_combined_params,
    set_density_grid_options,
)
from ecoscope.platform.tasks.config._meta_tasks import DensityGridOptions


def test_set_density_grid_options_defaults():
    opts = set_density_grid_options()
    assert isinstance(opts, DensityGridOptions)
    assert opts.opacity == 0.7
    assert opts.crs == "EPSG:3857"
    assert opts.intersecting_only is False
    assert opts.auto_scale_or_custom_cell_size is None


def test_density_grid_options_get_meshgrid_params():
    opts = set_density_grid_options(opacity=0.4, crs="EPSG:6933", intersecting_only=True)
    assert opts.get_meshgrid_params() == {
        "auto_scale_or_custom_cell_size": None,
        "crs": "EPSG:6933",
        "intersecting_only": True,
    }


def test_get_opacity_from_density_grid_options():
    opts = set_density_grid_options(opacity=0.25)
    assert get_opacity_from_combined_params(opts) == 0.25
