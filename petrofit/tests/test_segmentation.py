import os
import pytest

import numpy as np

from astropy.io import fits
from astropy.stats import sigma_clipped_stats

from photutils.segmentation import SegmentationImage

import petrofit as pf

from matplotlib import pyplot as plt


@pytest.fixture
def sersic_2d_image(data_dir):
    """fixture for Sersic 2D image """
    path = "sersic_2d_image.fits.gz"
    sersic_2d_path = os.path.join(data_dir, path)
    return fits.getdata(sersic_2d_path)


@pytest.fixture
def segm_and_cat(sersic_2d_image):
    """fixture for segmentation and catalog"""
    image_mean, image_median, image_stddev = sigma_clipped_stats(sersic_2d_image, sigma=3)
    threshold = image_stddev * 3
    npixels = 4 ** 2
    return pf.make_catalog(
        sersic_2d_image,
        threshold=threshold,
        deblend=True,
        npixels=npixels,
        contrast=0.00,
        plot=False)


def test_make_catalog(segm_and_cat, sersic_2d_image):
    """This tests the segmentation, deblending and catalog code"""
    cat, segm, segm_deblend = segm_and_cat

    # Check the number of sources detected
    assert len(cat) == 3

    # Check the type of the returned objects
    assert hasattr(cat, "to_table")
    assert isinstance(segm, SegmentationImage)
    assert isinstance(segm_deblend, SegmentationImage)

    # Check the properties of the segmentation maps
    assert np.all(segm.data >= 0)
    assert np.all(segm_deblend.data >= 0)

    # Ensure the segmentation maps have the same shape as the input image
    assert segm.shape == sersic_2d_image.shape
    assert segm_deblend.shape == sersic_2d_image.shape

    # Validate segmentation versus deblended segmentation
    assert not np.array_equal(segm.data, segm_deblend.data)  # They should be different if deblending is effective


def test_make_catalog_no_deblend(sersic_2d_image):
    image_mean, image_median, image_stddev = sigma_clipped_stats(sersic_2d_image, sigma=3)
    threshold = image_stddev * 3
    npixels = 4 ** 2
    cat, segm, segm_deblend = pf.make_catalog(
        sersic_2d_image,
        threshold=threshold,
        deblend=False,
        npixels=npixels,
        contrast=0.00,
        plot=True)
    plt.show()
    assert len(cat) == 1
    assert segm_deblend is None
    assert isinstance(segm, SegmentationImage)


def test_segmentation_functions(sersic_2d_image):
    """This tests the make_segments and deblend_segments functions"""

    image_mean, image_median, image_stddev = sigma_clipped_stats(sersic_2d_image, sigma=3)
    threshold = image_stddev * 3
    npixels = 4 ** 2

    # Testing make_segments
    segm = pf.make_segments(sersic_2d_image, npixels=npixels, threshold=threshold)

    assert isinstance(segm, SegmentationImage)
    assert segm.shape == sersic_2d_image.shape
    assert np.all(segm.data >= 0)
    assert len(np.unique(segm.data)) == 2  # account for background being labeled as 0

    # Testing deblend_segments
    segm_deblend = pf.deblend_segments(sersic_2d_image, segm, npixels=npixels, contrast=0.00)

    assert isinstance(segm_deblend, SegmentationImage)
    assert segm_deblend.shape == sersic_2d_image.shape
    assert np.all(segm_deblend.data >= 0)
    assert len(np.unique(segm_deblend.data)) >= len(np.unique(segm.data))


def test_masking_functions(sersic_2d_image, segm_and_cat):
    """This tests the segm_mask and masked_segm_image functions"""

    cat, segm, segm_deblend = segm_and_cat

    source_label = cat[0].label

    # Testing segm_mask
    mask = pf.segm_mask(source_label, segm_deblend)
    assert mask.shape == sersic_2d_image.shape
    unique_labels_in_mask = np.unique(segm_deblend.data[mask])
    assert set(unique_labels_in_mask) == {0, source_label}

    # Testing masked_segm_image
    masked_image = pf.masked_segm_image(source_label, sersic_2d_image, segm_deblend, fill=-9999)
    assert masked_image.shape == sersic_2d_image.shape
    assert np.all(masked_image[np.invert(mask)] == -9999)
    assert np.all(masked_image[mask] == sersic_2d_image[mask])


def test_get_functions(sersic_2d_image,segm_and_cat):
    """
    Test the convenience functions that
    get source properties
    """
    cat, segm, segm_deblend = segm_and_cat

    base_source = cat[0]

    source = base_source

    assert pf.get_source_position(source) == (base_source.maxval_xindex, base_source.maxval_yindex)
    assert pf.get_source_elong(source) == base_source.elongation.value
    assert pf.get_source_ellip(source) == base_source.ellipticity.value
    assert pf.get_source_theta(source) == base_source.orientation.to('rad').value

    x0, y0 = pf.get_source_position(source)
    ellip, theta = pf.get_source_ellip(source), pf.get_source_theta(source)

    assert np.round(pf.get_amplitude_at_r(200, sersic_2d_image, x0, y0 , ellip, theta), 6) == 0.036798


def test_plot_segments(sersic_2d_image, segm_and_cat):
    """Test segment plotting functions"""
    cat, segm, segm_deblend = segm_and_cat

    pf.plot_segments(segm, vmax=1, vmin=0)
    plt.show()

    pf.plot_segments(segm_deblend, vmax=1, vmin=0)
    plt.show()

    pf.plot_segment_residual(segm, sersic_2d_image, vmax=1, vmin=0)
    plt.show()


def test_source_photometry(sersic_2d_image, segm_and_cat):
    """
    Test the source_photometry function by doing photometry on
    a simulated image.
    """
    cat, segm, segm_deblend = segm_and_cat

    source = cat[0]

    max_pix = 120

    r_list = pf.make_radius_list(
        max_pix=max_pix,  # Max pixel to go up to
        n=max_pix  # the number of radii to produce
    )

    flux_arr, area_arr, error_arr = pf.source_photometry(
        # Inputs
        source,  # Source (`photutils.segmentation.catalog.SourceCatalog`)
        sersic_2d_image,  # Image as 2D array
        segm_deblend,  # Deblended segmentation map of image
        r_list,  # list of aperture radii

        # Options
        cutout_size=max(r_list)*2,  # Cutout out size, set to double the max radius
        bg_sub=True,  # Subtract background
        sigma=1, sigma_type='clip',  # Fit a 2D plane to pixels within 1 sigma of the mean
        plot=False, vmax=0, vmin=1,  # Show plot with max and min defined above
    )
    plt.show()


def test_order_cat(segm_and_cat):
    """Test if ordering is correct"""
    cat, segm, segm_deblend = segm_and_cat
    order = pf.order_cat(cat, 'area')

    source = None
    for idx in order:
        if source is not None:
            assert source.area > cat[idx].area

        source = cat[idx]
