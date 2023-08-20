import os
import numpy as np

import pytest

from scipy.interpolate import interp1d

from astropy.io import fits
from astropy.wcs import WCS
from astropy import units as u

import petrofit as pf


def test_natural_sort():
    test_list = ["item10", "item2", "item1a", "item1b", "item"]
    sorted_list = pf.natural_sort(test_list)
    expected_list = ["item", "item1a", "item1b", "item2", "item10"]
    assert sorted_list == expected_list


def test_flux_to_abmag():
    """Test HST Flux to AB mag conversions"""

    header = fits.Header()

    header['PHOTFLAM'] = 3.03865739999999e-20
    header['PHOTZPT'] = -21.10000000000001
    header['PHOTPLAM'] = 10551.04700000001

    mags = np.round(pf.hst_flux_to_abmag(100, header), 2)

    assert mags == 21.27


def test_make_radius_list():
    """test the make_radius_list function"""
    # Simple r_list
    max_pix = 100
    n = 50
    r_list = pf.make_radius_list(max_pix, n)
    assert max(r_list) == max_pix
    assert len(r_list) == n
    assert not np.any(r_list <= 0)

    # Odd division r_list
    max_pix = 100
    n = 30
    r_list = pf.make_radius_list(max_pix, n)
    assert max(r_list) == max_pix
    assert len(r_list) == n
    assert not np.any(r_list <= 0)

    # Log Space r_list
    max_pix = 100
    n = 8
    r_list = pf.make_radius_list(max_pix, n, log=True)
    assert max(r_list) == max_pix
    assert len(r_list) == n
    assert not np.any(r_list <= 0)
    expected = np.array([1., 1.93069773, 3.72759372, 7.19685673,
                         13.89495494, 26.82695795, 51.79474679, 100.])
    assert np.all(abs(r_list - expected)/expected < 0.01)


def test_match_catalogs():
    ra_1 = [10, 20, 30]
    dec_1 = [-10, -20, -30]
    ra_2 = [19.9, 9.9, 30.1]
    dec_2 = [-19.9, -10.1,  -29.9]

    idx, d2d, d3d = pf.match_catalogs(ra_1, dec_1, ra_2, dec_2)

    assert list(idx) == [1, 0, 2]
    assert all([d < 0.15 * u.deg for d in d2d])


def test_angular_to_pixel(data_dir):
    path = os.path.join(data_dir, "f105w_psf.fits.gz")
    header = fits.getheader(path)
    wcs = WCS(header)
    input_angular_r = 1 * u.arcsec
    pixel_r = pf.angular_to_pixel(input_angular_r, wcs)
    angular_r = pf.pixel_to_angular(pixel_r, wcs)
    angular_r == input_angular_r


def test_get_interpolated_values():
    # Sample data
    x = np.linspace(0, 10, 10)
    y = x ** 2

    # Using cubic interpolation
    x_new, y_new = pf.get_interpolated_values(x, y)

    # Check if returned values are the same for None kind
    x_none, y_none = pf.get_interpolated_values(x, y, kind=None)
    assert np.allclose(x, x_none)
    assert np.allclose(y, y_none)

    # Using cubic interpolation manually for verification
    f = interp1d(x, y, kind='cubic')
    x_check = np.linspace(min(x), max(x), num=5000, endpoint=True)
    y_check = f(x_check)

    assert len(x_new) == 5000
    assert np.allclose(y_new, y_check)

    # Case when input length is greater than default num
    x_large = np.linspace(0, 10, 6000)
    y_large = x_large ** 2
    x_new_large, y_new_large = pf.get_interpolated_values(x_large, y_large)

    assert len(x_new_large) == 6000


def test_closest_value_index():
    # Test for non-growing array
    array = np.array([5, 3, 2, 0, 3, 5])

    idx = pf.closest_value_index(1, array)
    assert idx == 2, "Expected 2, but got {}".format(idx)

    idx = pf.closest_value_index(10, array)
    assert idx == 0, "Expected 0, but got {}".format(idx)

    idx = pf.closest_value_index(-10, array)
    assert idx is None, "Expected None, but got {}".format(idx)

    # Test for growing array
    array = np.array([0, 2, 3, 5, 7, 10, 6, 1])
    idx = pf.closest_value_index(3.4, array, growing=True)
    assert idx == 2, "Expected 2, but got {}".format(idx)

    idx = pf.closest_value_index(-10, array, growing=True)
    assert idx == 0, "Expected 0, but got {}".format(idx)

    idx = pf.closest_value_index(100, array, growing=True)
    assert idx is None, "Expected None, but got {}".format(idx)


def test_ellipticity_and_elongation_conversion():
    # Test values
    ellip_values = [0, 0.25, 0.5, 0.75, 0.999]
    elong_values = [1, 4 / 3, 2, 4, 1000]

    for e, el in zip(ellip_values, elong_values):
        # Test ellip_to_elong conversion
        computed_elong = pf.ellip_to_elong(e)
        assert computed_elong == pytest.approx(el, 0.01)

        # Test elong_to_ellip conversion
        computed_ellip = pf.elong_to_ellip(el)
        assert computed_ellip == pytest.approx(e, 0.01)

    # Add a specific test for the upper boundary of ellip
    with pytest.raises(ZeroDivisionError):
        pf.ellip_to_elong(1)

    # Test for the lower boundary of elongation, it should not yield negative ellipticity
    assert pf.elong_to_ellip(1) == 0
