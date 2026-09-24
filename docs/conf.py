"""Sphinx configuration for the Winery Adventures documentation."""

import sys
from pathlib import Path

# Make the package importable by autodoc without installing it.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

project = "Winery Adventures"
copyright = "2026, Winery Adventures team"
author = "Winery Adventures team"

extensions = [
    "sphinx.ext.autodoc",  # documentation from the docstrings
    "sphinx.ext.napoleon",  # Google style docstrings (Args, Returns, Raises)
]

autodoc_member_order = "bysource"
exclude_patterns = ["_build"]
html_theme = "alabaster"
