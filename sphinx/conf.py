import os
import sys
sys.path.insert(0, os.path.abspath('../'))

# -- Project information -----------------------------------------------------
project = 'DeeProb-kit'
copyright = '2021, Lorenzo Loconte'
author = 'Lorenzo Loconte'
release = '0.6.0'

# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.githubpages',
]

exclude_patterns = ['docs', 'sphinx', 'examples', 'experiments', 'test']

# -- Options for HTML output -------------------------------------------------
html_theme = 'alabaster'
html_static_path = ['_static']
html_theme_options = {
    #'logo': 'deeprob-kit-logo.svg',
    #'logo_name': True,
    'github_user': 'loreloc',
    'github_repo': 'deeprob-kit',
    'github_button': True,
    'github_banner': True,
    'description': 'Python library for Deep Probabilistic Modeling',
}

# -- Other settings -----------------------------------------------------------
autodoc_member_order = 'bysource'
