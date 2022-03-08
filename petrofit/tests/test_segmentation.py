import pytest

from petrofit.segmentation import (
    plot_segments,
    plot_segment_residual,
    get_source_ellip,
    get_source_elong,
    get_source_position,
    get_source_theta,
    get_amplitude_at_r
)


def test_make_catalog(segm_and_cat):
    """This tests the segmentation, deblending and catalog code"""

    cat, segm, segm_deblend = segm_and_cat

    assert len(cat) == 3


def test_get_functions(sersic_2d_image,segm_and_cat):
    """
    Test the convenience functions that
    get source properties
    """
    cat, segm, segm_deblend = segm_and_cat

    base_source = cat[0]

    source = base_source

    assert get_source_position(source) == (base_source.maxval_xindex, base_source.maxval_yindex)
    assert get_source_elong(source) == base_source.elongation.value
    assert get_source_ellip(source) == base_source.ellipticity.value
    assert get_source_theta(source) == base_source.orientation.to('rad').value

    x0, y0 = get_source_position(source)
    ellip, theta = get_source_ellip(source), get_source_theta(source)

    assert get_amplitude_at_r(200, sersic_2d_image.data, x0, y0 , ellip, theta) == 0.036767338105531996


def test_plot_segments(sersic_2d_image, segm_and_cat):
    """Test segment plotting functions"""
    cat, segm, segm_deblend = segm_and_cat

    plot_segments(segm, vmax=1, vmin=0)

    plot_segments(segm_deblend, vmax=1, vmin=0)

    plot_segment_residual(segm, sersic_2d_image.data, vmax=1, vmin=0)


