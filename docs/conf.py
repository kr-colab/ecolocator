"""Sphinx configuration for the ecoLocator documentation."""

from __future__ import annotations

import sys
from datetime import datetime
from importlib import metadata
from pathlib import Path

# Make the package importable for autodoc even when the docs are built without
# first installing ecolocator (e.g. a quick local `sphinx-build`). On Read the
# Docs the package is installed via uv, so this is only a convenience fallback.
_PKG_SRC = Path(__file__).resolve().parent.parent / "ecolocator" / "src"
if _PKG_SRC.is_dir():
    sys.path.insert(0, str(_PKG_SRC))

# -- Project information ------------------------------------------------------

project = "ecoLocator"
author = "kr-colab"
copyright = f"{datetime.now():%Y}, {author}"

try:
    release = metadata.version("ecolocator")
except metadata.PackageNotFoundError:
    release = "0.1.0"
version = ".".join(release.split(".")[:2])

# -- General configuration ---------------------------------------------------

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx_copybutton",
    "sphinx_design",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- MyST (Markdown) ---------------------------------------------------------

myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "fieldlist",
    "linkify",
    "smartquotes",
    "substitution",
]
myst_heading_anchors = 3

# -- Autodoc / autosummary ---------------------------------------------------

autosummary_generate = True
autodoc_typehints = "description"
autodoc_member_order = "bysource"
autodoc_default_options = {
    "members": True,
    "show-inheritance": True,
}

# Docstrings in this project follow the NumPy convention.
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False

# -- Intersphinx -------------------------------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
}

# -- HTML output -------------------------------------------------------------

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_title = "ecoLocator"

html_theme_options = {
    "github_url": "https://github.com/kr-colab/ecolocator",
    "navigation_with_keys": True,
    "show_toc_level": 2,
}
