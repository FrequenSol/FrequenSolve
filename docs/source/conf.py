import os
import sys

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'FrequenSolve'
copyright = '2025, Jacob Badger'
author = 'Jacob Badger'
release = '0.1'

# -- Path setup --------------------------------------------------------------
# If your modules are in src/frequensolve, for example:
sys.path.insert(0, os.path.abspath("../../src"))

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",   # For Google/NumPy-style docstrings
    "myst_parser",           # If you use Markdown
]

napoleon_google_docstring = True
napoleon_numpy_docstring = False  # or True if you prefer NumPy-style
napoleon_include_init_with_doc = True

templates_path = ['_templates']

exclude_patterns = []

autodoc_typehints = "description"  # Show types in the parameter description instead of inline
autodoc_typehints_format = "short"  # Use short names for type annotations
autodoc_default_options = {
   'ignore-module-all': True,
   'undoc-members': False
}


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"

html_static_path = ["_static"]

html_css_files = [
   "custom.css",  # Relative to the _static directory
]

# HTML options
html_theme_options = {
   # Collapse navigation by default
   'collapse_navigation': True,
   # Maximum depth of top-level navigation
   'navigation_depth': 3,
   # Don't show deeper levels
   'titles_only': True
}

# TOC options
toc_object_entries = True
toc_object_entries_show_parents = 'hide'

# html_theme_options = {
#     "navigation_depth": 4,
#     "collapse_navigation": False,
#     "sticky_navigation": True,
#     "includehidden": True,
#     "titles_only": False,
# }
# html_logo = "images/logo.png"     # path to your logo
# html_favicon = "images/favicon.ico"  
