# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "nfflr"
copyright = "2024, Brian DeCost"
author = "Brian DeCost"
release = "0.1.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = []

templates_path = ["_templates"]
exclude_patterns = []


extensions = [
    "myst_nb",
    "numpydoc",
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    # "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosummary",
]
autosummary_generate = True
autodoc_inherit_docstrings = False
autodoc_member_order = "bysource"
autosummary_imported_members = False
autodoc_typehints = "signature"
napoleon_use_ivar = True

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output
html_css_files = ["https://pages.nist.gov/nist-header-footer/css/nist-combined.css"]

html_theme = "sphinx_book_theme"
html_theme_options = {
    "repository_url": "https://github.com/usnistgov/nfflr",
    "use_repository_button": True,
    "use_issues_button": True,
    "use_sidenotes": True,
    "use_download_button": False,
    "navigation_with_keys": False,
}
nb_execution_mode = "cache"

myst_enable_extensions = [
    "amsmath",
    # "colon_fence",
    "deflist",
    "dollarmath",
    "html_image",
]
myst_url_schemes = ("http", "https", "mailto")


# numpydoc config:
numpydoc_show_class_members = True
numpydoc_show_inherited_class_members = False
numpydoc_xref_param_type = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3.10", None),
    "torch": ("https://pytorch.org/docs/master/", None),
    "dgl": ("https://docs.dgl.ai/en/latest/", None),
}
