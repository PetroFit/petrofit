
.. image:: https://github.com/PetroFit/petrofit/raw/main/docs/images/petrofit_full_logo.png
    :width: 100%

PetroFit
--------
|CI tag| |rtd tag| |PyPI tag| |AJ tag| |zonodo tag| |python version tag| |repostatus tag| |astropy tag| |photutils tag| 

PetroFit is a package for calculating Petrosian properties, such as radii and concentration indices, as well as fitting
galaxy light profiles. In particular, PetroFit includes tools for performing accurate photometry, segmentations,
Petrosian profiling, and Sérsic fitting. Please see the `petrofit documentation <https://petrofit.readthedocs.io/en/latest/>`_
for installation instructions and a guide to the ``petrofit`` module.

.. image:: https://github.com/PetroFit/petrofit/raw/main/docs/images/multi_fit.png
    :width: 100%

Installation
------------

You can install PetroFit using ``pip install petrofit``. Please see
the `petrofit documentation <https://petrofit.readthedocs.io/en/latest/>`_ for detailed installation instructions.

Examples
--------

Please see the `petrofit documentation <https://petrofit.readthedocs.io/en/latest/>`_
for detailed examples, a quick stat guide, and instructions.

**Image Fitting**: 

PetroFit can be used with ``astropy`` models to fit psf convolved galaxy light profiles. Given a 2D image (``image``) and a psf (``PSF``), the following code snippet demonstrates how to fit a Sérsic model to the image:

.. code-block:: python

    import petrofit as pf
    from astropy.modeling import models

    sersic_model = models.Sersic2D(
            amplitude=1,
            r_eff=10,
            n=4,
            x_0=0, y_0=0,
            ellip=0.2,
            theta=25,
            bounds = pf.get_default_sersic_bounds(),
    )

    psf_sersic_model = pf.PSFConvolvedModel2D(
        sersic_model, psf=PSF, oversample=4, psf_oversample=1
    )

    fitted_model, fit_info = pf.fit_model(
        image, psf_sersic_model,
    )

**Photometry and Petrosian**: 

Given a 2D image (``image``) the following code snippet demonstrates how to create a Petrosian profile:

.. code-block:: python

    import petrofit as pf
    
    # Make a segmentation map and catalog using Photutils wrapper
    cat, segm, segm_deblend = pf.make_catalog(
        image,
        threshold=image.std()*3,
        wcs=None, deblend=True,
        npixels=npixels, nlevels=30, contrast=0.001,
    )

    # Photomerty on first source in catalog
    r_list = pf.make_radius_list(max_pix=50, n=50)
    flux_arr, area_arr, error_arr = pf.source_photometry(
        cat[0], # Source (`photutils.segmentation.catalog.SourceCatalog`)
        image, # Image as 2D array
        segm_deblend, # Deblended segmentation map of image
        r_list, # list of aperture radii
        cutout_size=max(r_list)*2, # Cutout out size, set to double the max radius
    )
    
    # Make a Petrosian profile 
    p = pf.Petrosian(r_list, area_arr, flux_arr)
    print("{:0.4f} pix".format(p.r_half_light))


Citation
--------

Please see the `petrofit documentation <https://petrofit.readthedocs.io/en/latest/>`_
for citation instructions. This information is also available in the `CITATION.rst`` 
file in the PetroFit repo.

License
-------

This project is Copyright (c) The PetroFit Team and licensed under
the terms of the BSD 3-Clause license. This package is based upon
the `Astropy package template <https://github.com/astropy/package-template>`_
which is licensed under the BSD 3-clause license. See the licenses folder for
more information.


Contributing
------------

We love contributions! petrofit is open source,
built on open source, and we'd love to have you hang out in our community.

**Imposter syndrome disclaimer**: We want your help. No, really.

There may be a little voice inside your head that is telling you that you're not
ready to be an open source contributor; that your skills aren't nearly good
enough to contribute. What could you possibly offer a project like this one?

We assure you - the little voice in your head is wrong. If you can write code at
all, you can contribute code to open source. Contributing to open source
projects is a fantastic way to advance one's coding skills. Writing perfect code
isn't the measure of a good developer (that would disqualify all of us!); it's
trying to create something, making mistakes, and learning from those
mistakes. That's how we all improve, and we are happy to help others learn.

Being an open source contributor doesn't just mean writing code, either. You can
help out by writing documentation, tests, or even giving feedback about the
project (and yes - that includes giving feedback about the contribution
process). Some of these contributions may be the most valuable to the project as
a whole, because you're coming to the project with fresh eyes, so you can see
the errors and assumptions that seasoned contributors have glossed over.

Note: This disclaimer was originally written by
`Adrienne Lowe <https://github.com/adriennefriend>`_ for a
`PyCon talk <https://www.youtube.com/watch?v=6Uj746j9Heo>`_, and was adapted by
petrofit based on its use in the README file for the
`MetPy project <https://github.com/Unidata/MetPy>`_.


.. |CI tag| image:: https://github.com/PetroFit/petrofit/actions/workflows/ci_tests.yml/badge.svg?branch=main
    :target: https://github.com/PetroFit/petrofit/actions/workflows/ci_tests.yml
    :alt: PetroFit CI status

.. |rtd tag| image:: https://readthedocs.org/projects/petrofit/badge/?version=latest
    :target: https://petrofit.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status

.. |PyPI tag| image:: https://img.shields.io/pypi/v/petrofit?color=blue
    :target: https://pypi.org/project/petrofit/
    :alt: PetroFit's PyPI Status

.. |AJ tag| image:: http://img.shields.io/badge/paper-AJ-blue.svg?style=flat
    :target: https://doi.org/10.3847/1538-3881/ac5908
    :alt: PetroFit AJ

.. |astropy tag| image:: http://img.shields.io/badge/powered%20by-Astropy-orange.svg?style=flat&colorB=D93F0B
    :target: https://pypi.org/project/astropy
    :alt: Powered by Astropy
    
.. |photutils tag| image:: http://img.shields.io/badge/powered%20by-Photutils-blue.svg?style=flat&colorB=084680
    :target: https://pypi.org/project/photutils/
    :alt: Powered by photutils

.. |zonodo tag| image:: http://img.shields.io/badge/zenodo-10.5281/zenodo.6386991-blue.svg?style=flat
    :target: https://zenodo.org/badge/latestdoi/348478663
    :alt: PetroFit Zenodo DOI

.. |repostatus tag| image:: https://www.repostatus.org/badges/latest/active.svg
   :alt: Project Status: Active – The project has reached a stable, usable state and is being actively developed.
   :target: https://www.repostatus.org/#active

.. |python version tag| image:: https://img.shields.io/python/required-version-toml?tomlFilePath=https%3A%2F%2Fraw.githubusercontent.com%2Fpetrofit%2Fpetrofit%2Fmain%2Fpyproject.toml
   :alt: Python Version from PEP 621 TOML
   :target: https://www.python.org/downloads/
