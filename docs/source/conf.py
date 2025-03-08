import datetime
import os
import sys

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "FrequenSolve"
copyright = f"{datetime.datetime.now().year}, Jacob Badger"
author = "Jacob Badger"
version = "0.1"  # Major + Minor version number
release = "0.1.1"  # Full version number

# -- Path setup --------------------------------------------------------------
# If your modules are in src/frequensolve, for example:
sys.path.insert(0, os.path.abspath("../../src"))

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",  # For Google/NumPy-style docstrings
    "sphinx.ext.viewcode",  # Add source code links
    "sphinx.ext.intersphinx",  # Link to other project's documentation
    "sphinx.ext.mathjax",  # Better math support
    "sphinx.ext.todo",  # Support for TODO items
    "myst_parser",  # If you use Markdown
    "sphinx_copybutton",  # Add copy button to code blocks
    "sphinx.ext.autosectionlabel",  # Allow referring to sections by name
]

napoleon_google_docstring = True
napoleon_numpy_docstring = False  # or True if you prefer NumPy-style
napoleon_include_init_with_doc = True

templates_path = ["_templates"]

exclude_patterns = []

autodoc_typehints = (
    "description"  # Show types in the parameter description instead of inline
)
autodoc_typehints_format = "short"  # Use short names for type annotations

# Intersphinx configuration
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"

html_static_path = ["_static"]

html_css_files = [
    "custom.css",  # Relative to the _static directory
]

html_js_files = [
    "custom.js",
]

# HTML options
html_theme_options = {
    "logo_only": False,
    "prev_next_buttons_location": "bottom",
    "style_external_links": True,
    "style_nav_header_background": "#2980B9",
    # Theme options
    "collapse_navigation": False,
    "sticky_navigation": True,
    "navigation_depth": 4,
    "includehidden": True,
    "titles_only": False,
}

# TOC options
toc_object_entries = True
toc_object_entries_show_parents = "hide"

# Sidebar logo
html_logo = "_static/logo.png"
html_favicon = "_static/favicon.ico"

# Footer configuration
html_show_sourcelink = True
html_show_sphinx = True
html_show_copyright = True

# Additional options
todo_include_todos = True
numfig = True  # Enable figure numbering
