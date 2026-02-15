# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

# -- Path setup --------------------------------------------------------------
# Add the project root so that autodoc can find the `puxle` package.
sys.path.insert(0, os.path.abspath(".."))

# -- Project information -----------------------------------------------------
project = "PuXle"
copyright = "2024, tinker495"
author = "tinker495"
release = "0.1.0"

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx_autodoc_typehints",
    "myst_parser",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Autodoc configuration ---------------------------------------------------
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
    "member-order": "bysource",
}
autodoc_typehints = "description"
autodoc_class_signature = "separated"

# Napoleon settings (Google-style docstrings)
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_attr_annotations = True

# MyST parser settings (for existing .md files)
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}
myst_enable_extensions = [
    "colon_fence",
    "deflist",
]

# Intersphinx mapping
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "jax": ("https://jax.readthedocs.io/en/latest/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}

# -- Options for HTML output -------------------------------------------------
html_theme = "furo"
html_title = "PuXle Documentation"
html_static_path = ["_static"]

html_theme_options = {
    "source_repository": "https://github.com/tinker495/PuXle",
    "source_branch": "main",
    "source_directory": "docs/",
}

# Suppress warnings for missing references to external types
nitpicky = False

# Mock imports for packages that may not be installed in the docs build env
autodoc_mock_imports = [
    "chex",
    "xtructure",
    "pddl",
    "termcolor",
    "tabulate",
    "cv2",
    "tqdm",
    "pandas",
]

# Prefer real JAX when available; fall back to mocks only when unavailable.
try:
    import jax  # noqa: F401
except Exception:
    autodoc_mock_imports.extend(["jax", "jaxlib"])
