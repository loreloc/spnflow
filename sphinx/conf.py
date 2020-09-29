# -- Project information -----------------------------------------------------

project = 'SPNFlow'
copyright = '2020, Lorenzo Loconte'
author = 'Lorenzo Loconte'
release = '0.4.0'

# -- General configuration ---------------------------------------------------

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.githubpages',
]

exclude_patterns = ['docs', 'examples', 'experiments']

# -- Options for HTML output -------------------------------------------------

html_theme = 'alabaster'
html_static_path = ['_static']
html_theme_options = {
    'logo': 'spnflow-logo.svg',
    'logo_name': True,
    'github_user': 'loreloc',
    'github_repo': 'spnflow',
    'github_button': True,
    'github_banner': True,
    'description': 'Sum-Product Networks and Normalizing Flows for Tractable Density Estimation',

}

# -- Other settings -----------------------------------------------------------
autodoc_member_order = 'bysource'
