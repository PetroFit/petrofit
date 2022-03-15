import os
import pytest

import numpy as np

from astropy.stats import sigma_clipped_stats
from astropy.nddata import CCDData
from astropy.modeling import models
from astropy.io import fits
from astropy.wcs import WCS


from petrofit.segmentation import make_catalog
from petrofit.petrosian import Petrosian
from petrofit.fitting import model_to_image
from petrofit.photometry import (
    make_radius_list,
    source_photometry,
    order_cat
)


def make_image():
    """Make 2D Sersic2D image"""
    imsize = 500

    d = 5

    sersic_model = models.Sersic2D(
        amplitude=1,
        r_eff=25,
        n=2,
        x_0=imsize / 2,
        y_0=imsize / 2,
        ellip=0,
        theta=0,
        bounds={
            'amplitude': (0., None),
            'r_eff': (0, None),
            'n': (0, 10),
            'ellip': (0, 1),
            'theta': (-2 * np.pi, 2 * np.pi),
        },
    )

    sersic_model += models.Sersic2D(
        amplitude=1,
        r_eff=25,
        n=2,
        x_0=imsize * 0.75,
        y_0=imsize * 0.75,
        ellip=0.5,
        theta=np.pi / 4,
        bounds={
            'amplitude': (0., None),
            'r_eff': (0, None),
            'n': (0, 10),
            'ellip': (0, 1),
            'theta': (-2 * np.pi, 2 * np.pi),
        },
    )

    sersic_model += models.Sersic2D(
        amplitude=1,
        r_eff=25,
        n=1,
        x_0=imsize * 0.25,
        y_0=imsize * 0.25,
        ellip=0.2,
        theta=np.pi / 6,
        bounds={
            'amplitude': (0., None),
            'r_eff': (0, None),
            'n': (0, 10),
            'ellip': (0, 1),
            'theta': (-2 * np.pi, 2 * np.pi),
        },
    )

    model_image = model_to_image(sersic_model, imsize)

    image_mean, image_median, image_stddev = sigma_clipped_stats(model_image, sigma=3)

    # model_image += np.random.normal(0, 3*image_stddev, size=model_image.shape)

    wcs = WCS()

    header = wcs.to_header()

    for param_name, param_val in zip(sersic_model.param_names, sersic_model.parameters):
        header[param_name] = param_val

    header['BUNIT'] = 'electron / s '

    path = "sersic_2d_image.fits.gz"
    sersic_2d_path = os.path.join(os.path.dirname(__file__), path)
    fits.writeto(
        sersic_2d_path,
        data=model_image,
        header=header,
        overwrite=True
    )



@pytest.fixture
def sersic_2d_image():
    """fixture for Sersic 2D image """
    path =  "sersic_2d_image.fits.gz"
    sersic_2d_path =  os.path.join(os.path.dirname(__file__), path)

    if not os.path.isfile(sersic_2d_path):
        make_image()

    return CCDData.read(sersic_2d_path)


@pytest.fixture
def segm_and_cat(sersic_2d_image):
    """fixture for segmentation and catalog"""
    image_mean, image_median, image_stddev = sigma_clipped_stats(sersic_2d_image, sigma=3)

    threshold = image_stddev * 3

    # Define smoothing kernel
    kernel_size = 3
    fwhm = 3

    # Min Source size (area)
    npixels = 4 ** 2

    return make_catalog(
        sersic_2d_image,
        threshold=threshold,
        deblend=True,
        kernel_size=kernel_size,
        fwhm=fwhm,
        npixels=npixels,
        contrast=0.00,
        plot=False,
    )



@pytest.fixture
def sersic_petrosian(sersic_2d_image, segm_and_cat):
    """fixture for Sersic model image Petrosian"""
    cat, segm, segm_deblend = segm_and_cat

    source = cat[order_cat(cat)[0]]

    max_pix = 200

    r_list = make_radius_list(
        max_pix=max_pix,  # Max pixel to go up to
        n=max_pix  # the number of radii to produce
    )

    flux_list, area_list, error_arr = source_photometry(

        # Inputs
        source,  # Source (`photutils.segmentation.catalog.SourceCatalog`)
        sersic_2d_image.data,  # Image as 2D array
        segm_deblend,  # Deblended segmentation map of image
        r_list,  # list of aperture radii

        # Options
        cutout_size=max(r_list) * 2,  # Cutout out size, set to double the max radius
        bkg_sub=False,  # Subtract background
        sigma=1, sigma_type='clip',  # Fit a 2D plane to pixels within 1 sigma of the mean
        plot=False, vmax=0, vmin=1,  # Show plot with max and min defined above
    )

    return Petrosian(r_list, area_list, flux_list)
