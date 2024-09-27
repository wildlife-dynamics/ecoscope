# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#

# -- Project information -----------------------------------------------------

import os
import sys
import subprocess

sys.path.insert(0, os.path.abspath("../.."))  # Necessary for viewcode
sys.path.insert(0, os.path.abspath(".."))

project = "Ecocope"
copyright = "2022, Wildlife Dynamics"
author = "Wildlife Dynamics"

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "autoapi.extension",
    "nbsphinx",
    "pydata_sphinx_theme",
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "myst_parser",
]

myst_enable_extensions = [
    "colon_fence",
]

nbsphinx_execute = "never"

master_doc = "autoapi/ecoscope/index"
master_doc = "index"

autoapi_type = "python"
autoapi_dirs = ["../../ecoscope"]
autoapi_ignore = ["*/contrib/*"]

autodoc_typehints = "description"

# autoapi_keep_files = True
autoapi_options = [
    "members",
    "undoc-members",
    "private-members",
    "show-inheritance",
    "special-members",
    "imported-members",
]

# Add any paths that contain templates here, relative to this directory.
# templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
html_theme = "pydata_sphinx_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]


html_title = "Ecoscope Docs"
html_favicon = "_static/images/favicon.ico"
html_logo = "_static/images/logo.svg"

# material theme options (see theme.conf for more information)
html_theme_options = {
    "github_url": "https://github.com/wildlife-dynamics/ecoscope",
    "pygment_light_style": "rainbow_dash",
    "pygment_dark_style": "dracula",
}

commit_id = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).strip().decode("ascii")
nbsphinx_prolog = f"""
.. image:: https://colab.research.google.com/assets/colab-badge.svg
  :target: {{{{ "https://githubtocolab.com/wildlife-dynamics/ecoscope/blob/{commit_id}/doc/source/" + env.docname + ".ipynb" }}}}

----
"""  # noqa
