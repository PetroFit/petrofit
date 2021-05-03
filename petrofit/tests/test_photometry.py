
import numpy as np

from astropy.io import fits

from petrofit.photometry import (
    radial_elliptical_aperture,
    radial_elliptical_annulus,
    make_radius_list,
    photometry_step,
    source_photometry,
    flux_to_abmag,
    order_cat
)


def test_radial_elliptical_aperture():
    """Test if loading elliptical aperture"""
    radial_elliptical_aperture(position=(0, 0), r=10)


def test_radial_elliptical_annulus():
    """Test if loading elliptical annulus"""
    radial_elliptical_annulus(position=(0, 0), r=10, dr=5)


def test_make_radius_list():
    """test the make_radius_list function"""
    # Simple r_list
    max_pix = 100
    n = 50

    r_list = make_radius_list(max_pix, n)

    assert max(r_list) == max_pix
    assert len(r_list) == n
    assert not np.any(r_list <= 0)

    # Odd division r_list
    max_pix = 100
    n = 30

    r_list = make_radius_list(max_pix, n)

    assert max(r_list) == max_pix
    assert len(r_list) == n
    assert not np.any(r_list <= 0)


def test_photometry_step():
    """
    Test photometry core by doing photometry
    where the total flux = pixel area.
    """
    image_size = 30
    ones_image = np.ones((30, 30))

    r_list = make_radius_list(image_size // 2, image_size // 2)

    results = photometry_step(
        np.array(ones_image.shape) // 2,
        r_list,
        ones_image,
        plot=True
    )

    results = [r_list] + list(results)

    results = [np.round(l, 6) for l in results]

    for r, photometry, aperture_area, error in zip(*results):
        area = np.round(np.pi * r ** 2, 6)

        photo_error = abs(photometry - area) / area
        assert photo_error < 0.01, "{}".format(photo_error)

        area_error = abs(aperture_area - area) / area
        assert area_error < 0.01, "{}".format(area_error)


def test_source_photometry(sersic_2d_image, segm_and_cat):
    """
    Test the source_photometry function by doing photometry on
    a simulated image.
    """
    cat, segm, segm_deblend = segm_and_cat

    source = cat[0]

    max_pix = 120

    r_list = make_radius_list(
        max_pix=max_pix, # Max pixel to go up to
        n=max_pix # the number of radii to produce
    )


    flux_arr, area_arr, error_arr = source_photometry(

        # Inputs
        source, # Source (`photutils.segmentation.catalog.SourceCatalog`)
        sersic_2d_image.data, # Image as 2D array
        segm_deblend, # Deblended segmentation map of image
        r_list, # list of aperture radii

        # Options
        cutout_size=max(r_list)*2, # Cutout out size, set to double the max radius
        bkg_sub=True, # Subtract background
        sigma=1, sigma_type='clip', # Fit a 2D plane to pixels within 1 sigma of the mean
        plot=True, vmax=0, vmin=1, # Show plot with max and min defined above
    )



def test_order_cat(segm_and_cat):
    """Test if ordering is correct"""
    cat, segm, segm_deblend = segm_and_cat
    order = order_cat(cat, 'area')

    source = None
    for idx in order:
        if source is not None:
            assert source.area > cat[idx].area

        source = cat[idx]


def test_flux_to_abmag():
    """Test HST Flux to AB mag conversions"""

    header = fits.Header()

    header['PHOTFLAM'] = 3.03865739999999e-20
    header['PHOTZPT'] =  -21.10000000000001
    header['PHOTPLAM'] = 10551.04700000001

    mags = np.round(flux_to_abmag(100, header), 2)

    assert mags == 21.27
