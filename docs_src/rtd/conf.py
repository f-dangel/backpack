# Sphinx Configuration file
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# http://www.sphinx-doc.org/en/master/config

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))

# -- Project information -----------------------------------------------------

project = "BackPACK"
copyright = "2019, F. Dangel, F. Kunstner"
author = "F. Dangel, F. Kunstner"

# The full version, including alpha/beta/rc tags
release = "1.2.0"
master_doc = "index"

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.coverage",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.intersphinx",
    "sphinx_gallery.gen_gallery",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

# -- Intersphinx config -----------------------------------------------------

intersphinx_mapping = {"torch": ("https://pytorch.org/docs/stable/", None)}

# -- Sphinx Gallery config ---------------------------------------------------

sphinx_gallery_conf = {
    "examples_dirs": [
        "../examples/basic_usage",
        "../examples/use_cases",
    ],  # path to your example scripts
    "gallery_dirs": [
        "basic_usage",
        "use_cases",
    ],  # path to where to save gallery generated output
    "default_thumb_file": "assets/backpack_logo_torch.png",
    "filename_pattern": "example",
}

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
# html_theme = "sphinx_rtd_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
# html_static_path = [""]
