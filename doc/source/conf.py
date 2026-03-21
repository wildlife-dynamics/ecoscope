# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to show here.
#

# -- Project information -----------------------------------------------------

import os
import subprocess
import sys

sys.path.insert(0, os.path.abspath("../.."))  # Necessary for viewcode
sys.path.insert(0, os.path.abspath(".."))

project = "Ecoscope"
copyright = "2025, Wildlife Dynamics"
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


html_title = "Ecoscope Documentation"
html_favicon = "_static/images/favicon.ico"
html_logo = "_static/images/logo.svg"

# Modern pydata-sphinx-theme configuration for responsive, accessible design
html_theme_options = {
    "logo": {
        "text": "Ecoscope",
        "image_light": "_static/images/logo.svg",
        "image_dark": "_static/images/logo.svg",
    },
    "header_links_before_dropdown": 7,
    "navbar_align": "left",
    "show_nav_level": 2,
    "navbar_center": ["navbar-nav"],
    "navbar_end": ["navbar-icon-links"],
    "github_url": "https://github.com/wildlife-dynamics/ecoscope",
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/wildlife-dynamics/ecoscope",
            "icon": "fab fa-github-square",
        },
    ],
    "pygments_light_style": "rainbow_dash",
    "pygments_dark_style": "dracula",
    "use_edit_page_button": True,
    "announcement": "🌍 Conservation Data Analytics for Wildlife Researchers",
}

commit_id = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).strip().decode("ascii")
nbsphinx_prolog = f"""
.. image:: https://colab.research.google.com/assets/colab-badge.svg
  :target: {{{{ "https://colab.research.google.com/github/wildlife-dynamics/ecoscope/blob/{commit_id}/doc/source/" + env.docname|urlencode + ".ipynb" }}}}

----
"""  # noqa

# Custom CSS for enhanced styling
html_css_files = [
    "custom.css",
]
