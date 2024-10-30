# -*- coding: utf-8 -*-
# Licensed under a 3-clause BSD style license - see LICENSE.rst

import os
import sys
import datetime
from importlib import import_module

try:
    from sphinx_astropy.conf.v1 import *  # noqa
except ImportError:
    print('ERROR: the documentation requires the sphinx-astropy package to be installed')
    sys.exit(1)

# Get configuration information from setup.cfg
import tomllib

with open(os.path.join(os.path.dirname(__file__), '..', 'pyproject.toml'), "rb") as f:
    setup_cfg = tomllib.load(f)

# -- General configuration ----------------------------------------------------

highlight_language = 'python3'
exclude_patterns.append('_templates')
exclude_patterns.append('images/*ipynb')
rst_epilog += """
"""
extensions += [
    'nbsphinx',
    'sphinx_automodapi.automodapi',
    'sphinx_automodapi.smart_resolver'
]
nbsphinx_execute = 'auto'

# -- Project information ------------------------------------------------------

project = "PetroFit"
author = setup_cfg["project"]['authors'][0]['name']
module_name = setup_cfg["project"]['name']
copyright = '{0}, {1}'.format(
    datetime.datetime.now().year, author)
import_module(module_name)
package = sys.modules[module_name]
release = package.__version__
version = release
if "dev" in release:
    version = package.__version__.split('dev', 1)[0] + 'dev'

# -- Options for HTML output --------------------------------------------------

html_theme = "sphinx_rtd_theme"
html_logo = "./images/petrofit_logo_no_bg.png"
html_favicon = './images/petrofit_logo.ico'
html_theme_options = {
    'style_nav_header_background': 'linear-gradient(90deg, #00B0F0 0%, #8D00ED 100%)',
}
html_static_path = ['_static']
html_css_files = ["css/custom.css"]
html_title = '{0} v{1}'.format(project, release)
htmlhelp_basename = project + 'doc'
modindex_common_prefix = ["petrofit."]

# -- Options for LaTeX output -------------------------------------------------

latex_documents = [('index', project + '.tex', project + u' Documentation',
                    author, 'manual')]

# -- Options for manual page output -------------------------------------------

man_pages = [('index', project.lower(), project + u' Documentation',
              [author], 1)]

# -- Resolving issue number to links in changelog -----------------------------

github_issues_url = setup_cfg['project']['urls']['Bug Tracker']


# -- Options for linkcheck output -------------------------------------------

linkcheck_retry = 5
linkcheck_ignore = [
    r'https://github\.com/robelgeda/petrofit/(?:issues|pull)/\d+',
]
linkcheck_timeout = 180
linkcheck_anchors = False
