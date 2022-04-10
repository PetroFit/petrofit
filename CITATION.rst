##################
Citing and Credits
##################

Publications
============

|ApJ tag| |zonodo tag| |ascl tag|

If you would like to cite PetroFit for a publication, please use the acknowledgments below.
Given the intimate relationship between PetroFit and Photutils, we ask that you also cite
Photutils:

.. code-block:: text

    This research made use of Photutils, an Astropy package for
    detection and photometry of astronomical sources (Bradley et al.
    20XX). This research made use of PetroFit (Geda et al. 2022), a
    package based on Photutils, for calculating Petrosian properties
    and fitting galaxy light profiles.

Where `(Bradley et al. 20XX)` is a citation to the `Zenodo record of the Photutils version
<https://doi.org/10.5281/zenodo.596036>`_
that was used. For more information on how to cite Photutils or get an example BibTeX entry, please visit
the `Photutils documentation <https://photutils.readthedocs.io/en/stable/citation.html>`_.
If you need to cite a specific version of PetroFit, please use
`PetroFit's Zonodo record <https://zenodo.org/badge/latestdoi/348478663>`_
in addition to the acknowledgments above.

*Geda et al. 2022* BibTeX (`download .bib file <https://iopscience.iop.org/export?articleId=1538-3881/163/5/202&doi=10.3847/1538-3881/ac5908&exportFormat=iopexport_bib&exportType=abs&navsubmit=Export+abstract>`_):

.. code-block:: text

    @article{Geda_2022,
        doi = {10.3847/1538-3881/ac5908},
        url = {https://doi.org/10.3847/1538-3881/ac5908},
        year = 2022,
        month = {apr},
        publisher = {American Astronomical Society},
        volume = {163},
        number = {5},
        pages = {202},
        author = {Robel Geda and Steven M. Crawford and Lucas Hunt and Matthew Bershady and Erik Tollerud and Solohery Randriamampandry},
        title = {{PetroFit}: A Python Package for Computing Petrosian Radii and Fitting Galaxy Light Profiles},
        journal = {The Astronomical Journal},
        abstract = {PetroFit is an open-source Python package based on Astropy and Photutils that can calculate Petrosian profiles and fit galaxy images. It offers end-to-end tools for making accurate photometric measurements, estimating morphological properties, and fitting 2D models to galaxy images. Petrosian metric radii can be used for model parameter estimation and aperture photometry to provide accurate total fluxes. Correction tools are provided for improving Petrosian radii estimates affected by galaxy morphology. PetroFit also provides tools for sampling Astropy-based models (including custom profiles and multicomponent models) onto image grids and enables point-spread function convolution to account for the effects of seeing. These capabilities provide a robust means of modeling and fitting galaxy light profiles. We have made the PetroFit package publicly available on GitHub ( PetroFit/petrofit ) and PyPi (pip install petrofit).}
    }

PetroFit Badges
===============

If a significant portion of your code uses PetroFit, please show support by adding the
following badge to your GitHub `README` file:

PetroFit Badge:

|petrofit tag|

reStructuredText:

.. code-block:: rst

    |petrofit tag|

    .. |petrofit tag| image:: https://img.shields.io/badge/powered%20by-PetroFit-blue.svg?style=flat&color=00B0F0
        :target: https://github.com/PetroFit/petrofit
        :alt: PetroFit

Markdown:

.. code-block:: text

    [![PetroFit](https://img.shields.io/badge/powered%20by-PetroFit-blue.svg?style=flat&color=00B0F0)](https://github.com/PetroFit/petrofit)

HTML:

.. code-block:: html

    <a href="https://github.com/PetroFit/petrofit"><img src="https://img.shields.io/badge/powered%20by-PetroFit-blue.svg?style=flat&color=00B0F0" alt="PetroFit"></a>
f

.. |ApJ tag| image:: http://img.shields.io/badge/ApJ-10.3847/1538-%2D3881/ac5908-blue.svg?style=flat
    :target: https://doi.org/10.3847/1538-3881/ac5908
    :alt: PetroFit ApJ

.. |ads tag| image:: http://img.shields.io/badge/ADS-2022arXiv220213493G-blue.svg?style=flat
    :target: https://ui.adsabs.harvard.edu/abs/2022arXiv220213493G/abstract
    :alt: PetroFit ADS

.. |zonodo tag| image:: http://img.shields.io/badge/zenodo-10.5281/zenodo.6386991-blue.svg?style=flat
    :target: https://zenodo.org/badge/latestdoi/348478663
    :alt: PetroFit Zenodo DOI

.. |arxiv tag| image:: http://img.shields.io/badge/arxiv-2202.13493-blue.svg?style=flat&colorB=b31a1a
    :target: https://arxiv.org/abs/2202.13493
    :alt: PetroFit arxiv

.. |ascl tag| image:: https://img.shields.io/badge/ascl-2203.013-black.svg?colorB=262255
    :target: https://ascl.net/2203.013
    :alt: ascl:2203.013

.. |petrofit tag| image:: https://img.shields.io/badge/powered%20by-PetroFit-blue.svg?style=flat&color=00B0F0
    :target: https://github.com/PetroFit/petrofit
    :alt: PetroFit