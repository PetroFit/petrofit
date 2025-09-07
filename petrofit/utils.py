import re

import numpy as np

from scipy.interpolate import interp1d

from astropy.wcs.utils import proj_plane_pixel_scales
from astropy import units as u
from astropy.coordinates import SkyCoord

__all__ = [
    "match_catalogs",
    "angular_to_pixel",
    "pixel_to_angular",
    "elliptical_area_to_r",
    "circle_area_to_r",
    "get_interpolated_values",
    "closest_value_index",
    "cutout_subtract",
    "hst_flux_to_abmag",
    "make_radius_list",
    "natural_sort",
    "ellip_to_elong",
    "elong_to_ellip",
]


def natural_sort(l):
    """Sort the given list of strings or numbers without needing leading zeros"""
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split("([0-9]+)", key)]
    return sorted(l, key=alphanum_key)


def ellip_to_elong(ellip):
    """Convert ellipticity to elongation"""
    return 1 / (1 - ellip)


def elong_to_ellip(elong):
    """Convert elongation to ellipticity"""
    return (elong - 1) / elong


def make_radius_list(max_pix, n, log=False):
    """Make an array of radii of size n up to max_pix"""
    if log:
        return np.logspace(
            0, np.log10(max_pix), num=n, endpoint=True, base=10.0, dtype=float, axis=0
        )
    else:
        return np.array([x * max_pix / n for x in range(1, n + 1)])


def hst_flux_to_abmag(flux, header):
    """Convert HST flux to AB Mag"""
    if not type(flux) in [int, float]:
        flux = np.array(flux)
        flux[np.where(flux <= 0)] = np.nan
    elif flux <= 0:
        return np.nan

    PHOTFLAM = header["PHOTFLAM"]
    PHOTZPT = header["PHOTZPT"]
    PHOTPLAM = header["PHOTPLAM"]

    STMAG_ZPT = (-2.5 * np.log10(PHOTFLAM)) + PHOTZPT
    ABMAG_ZPT = STMAG_ZPT - (5.0 * np.log10(PHOTPLAM)) + 18.692

    return -2.5 * np.log10(flux) + ABMAG_ZPT


def match_catalogs(ra_1, dec_1, ra_2, dec_2, unit="deg"):
    """Wrapper for `SkyCoord.match_to_catalog_sky`"""
    cat1_coords = SkyCoord(ra=ra_1, dec=dec_1, unit=unit)
    cat2_coords = SkyCoord(ra=ra_2, dec=dec_2, unit=unit)
    return cat1_coords.match_to_catalog_sky(cat2_coords)


def angular_to_pixel(angular_diameter, wcs):
    """Convert angular angular diameter to pixel width using wcs"""
    pixel_scales = proj_plane_pixel_scales(wcs)
    assert np.allclose(*pixel_scales)
    pixel_scale = pixel_scales[0] * wcs.wcs.cunit[0] / u.pix

    pixel_size = angular_diameter / pixel_scale.to(angular_diameter.unit / u.pix)
    pixel_size = pixel_size.value

    return pixel_size


def pixel_to_angular(pixel_size, wcs):
    """Convert angular pixel width to angular diameter using wcs"""
    pixel_scales = proj_plane_pixel_scales(wcs)
    assert np.allclose(*pixel_scales)
    pixel_scale = pixel_scales[0] * wcs.wcs.cunit[0] / u.pix

    if not hasattr(pixel_size, "unit"):
        pixel_size = pixel_size * u.pix

    angular_diameter = pixel_size * pixel_scale.to(u.arcsec / u.pix)
    return angular_diameter


def elliptical_area_to_r(area, elong):
    """Convert elliptical area to radius by providing elongation"""
    a = np.sqrt(elong * area / (np.pi))
    b = a / elong
    return a, b


def circle_area_to_r(area):
    """Convert circular area to radius"""
    return np.sqrt(area / np.pi)


def get_interpolated_values(x, y, num=5000, kind="cubic"):
    """
    Interpolate values to a new grid

    Parameters
    ----------
    x : array like
        x values
    y : array like
        y values
    num : int, optional
        Number of points to interpolate to. Default is 5000.
    kind : str, optional
        Kind of interpolation. Default is 'cubic'.

    Returns
    -------
    x_new, y_new : array like
        Interpolated values
    """
    if kind is None:
        return x, y

    if len(x) > num:
        num = len(x)

    f = interp1d(x, y, kind=kind)
    x_new = np.linspace(min(x), max(x), num=num, endpoint=True)
    y_new = f(x_new)
    return x_new, y_new


def closest_value_index(value, array, growing=False):
    """Return first index closes to value"""

    if not growing:
        idx_list = np.where(array <= value)[0]
    elif growing:
        idx_list = np.where(array >= value)[0]

    idx = None
    if idx_list.size > 0:
        idx = idx_list[0]
        idx = abs(array[: idx + 1] - value).argmin()
    return idx

def cutout_subtract(image, target, x, y):
    """
    Subtract cutout from image

    Parameters
    ----------
    image : array like
        Main image

    target : array like
        Cutout image

    x, y : int
        Center to subtract from

    Returns
    -------
    Copied array
        subtracted
    """

    dy, dx = target.shape
    bounds = np.array([y - dy // 2, y + dy // 2, x - dx // 2, x + dx // 2])
    bounds[bounds < 0] = 0
    ymin, ymax, xmin, xmax = bounds
    image[ymin:ymax, xmin:xmax] -= target
    return image
