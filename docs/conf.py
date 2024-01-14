# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

# note: develop with `sphinx-autobuild docs/source docs/_build/html -c docs`
project = "NFFLr"
copyright = "2023, Brian DeCost"
author = "Brian DeCost"
# release = "0.1.0"

# The default replacements for |version| and |release|, also used in various
# other places throughout the built documents.
#
import re
import nfflr

# The short X.Y version (including .devXXXX, rcX, b1 suffixes if present)
version = re.sub(r"(\d+\.\d+)\.\d+(.*)", r"\1\2", nfflr.__version__)
version = re.sub(r"(\.dev\d+).*?$", r"\1", version)
# The full version, including alpha/beta/rc tags.
release = nfflr.__version__
print("%s %s" % (version, release))


# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "nbsphinx",
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosummary",
    # "autoapi.extension",
]

templates_path = ["source/_templates"]
exclude_patterns = []

html_css_files = ["https://pages.nist.gov/nist-header-footer/css/nist-combined.css"]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_book_theme"
html_theme_options = {
    "repository_url": "https://github.com/usnistgov/nfflr",
    "use_repository_button": True,
    "use_issues_button": True,
}
html_static_path = ["source/_static"]

# html_theme = "alabaster"
# html_theme_options = {
#     # 'logo': 'logo.png',
#     "github_user": "usnistgov",
#     "github_repo": "nfflr",
#     "github_button": True,
#     "github_banner": False,
#     "sidebar_width": "220px",
#     "page_width": "1080px"
# }

# autodoc stuff
# Disable docstring inheritance
# autodoc_inherit_docstrings = False

# generate individual pages for methods/functions
autosummary_generate = True
autodoc_typehints = "description"
napoleon_use_ivar = True

# autoapi_dirs = ["../../nfflr"]
# autoapi_type = "python"

# autoapi_options = [
#     "members",
#     "undoc-members",
#     "show-inheritance",
#     "show-module-summary",
#     "imported-members",
# ]

# def skip_member(app, what, name, obj, skip, options):
#     # skip submodules
#     if what == "module":
#         skip = True
#     return skip

# def setup(sphinx):
#     sphinx.connect("autoapi-skip-member", skip_member)
