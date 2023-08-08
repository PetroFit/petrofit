import re

import numpy as np

from scipy.interpolate import interp1d

from astropy.stats import gaussian_sigma_to_fwhm
from astropy.wcs.utils import proj_plane_pixel_scales
from astropy import units as u
from astropy.coordinates import SkyCoord

from matplotlib import pyplot as plt

from .modeling.fitting import fit_gaussian2d, plot_fit

__all__ = [
    'match_catalogs', 'angular_to_pixel', 'pixel_to_angular',
    'elliptical_area_to_r', 'circle_area_to_r', 'get_interpolated_values',
    'closest_value_index', 'plot_target', 'cutout_subtract',
    'measure_fwhm', 'hst_flux_to_abmag', 'natural_sort'
]


def natural_sort(l):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    return sorted(l, key = alphanum_key)


def hst_flux_to_abmag(flux, header):
    """Convert HST flux to AB Mag"""
    if not type(flux) in [int, float]:
        flux = np.array(flux)
        flux[np.where(flux <= 0)] = np.nan
    elif flux <= 0:
        return np.nan

    PHOTFLAM = header['PHOTFLAM']
    PHOTZPT = header['PHOTZPT']
    PHOTPLAM = header['PHOTPLAM']

    STMAG_ZPT = (-2.5 * np.log10(PHOTFLAM)) + PHOTZPT
    ABMAG_ZPT = STMAG_ZPT - (5. * np.log10(PHOTPLAM)) + 18.692

    return -2.5 * np.log10(flux) + ABMAG_ZPT


def match_catalogs(ra_1, dec_1, ra_2, dec_2, unit='deg'):
    """Wrapper for `SkyCoord.match_to_catalog_sky`"""
    cat1_coords = SkyCoord(ra=ra_1, dec=dec_1, unit=unit)
    cat2_coords = SkyCoord(ra=ra_2, dec=dec_2, unit=unit)
    return cat1_coords.match_to_catalog_sky(cat2_coords)


def angular_to_pixel(angular_diameter, wcs):
    pixel_scales = proj_plane_pixel_scales(wcs)
    assert np.allclose(*pixel_scales)
    pixel_scale = pixel_scales[0] * wcs.wcs.cunit[0] / u.pix

    pixel_size = angular_diameter / pixel_scale.to(angular_diameter.unit / u.pix)
    pixel_size = pixel_size.value

    return pixel_size


def pixel_to_angular(pixel_size, wcs):
    pixel_scales = proj_plane_pixel_scales(wcs)
    assert np.allclose(*pixel_scales)
    pixel_scale = pixel_scales[0] * wcs.wcs.cunit[0] / u.pix

    if not hasattr(pixel_size, 'unit'):
        pixel_size = pixel_size * u.pix

    angular_diameter = pixel_size * pixel_scale.to(u.arcsec / u.pix)
    return angular_diameter


def elliptical_area_to_r(area, elong):
    a = np.sqrt(elong * area / (np.pi))
    b = a / elong
    return a, b


def circle_area_to_r(area):
    return np.sqrt(area / (np.pi))


def get_interpolated_values(x, y, num=5000, kind='cubic'):
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
        idx = abs(array[:idx + 1] - value).argmin()
    return idx


def plot_target(position, image, size, vmin=None, vmax=None):
    x, y = position
    if not isinstance(image, np.ndarray):
        image = image.data
    plt.imshow(image, vmin=vmin, vmax=vmax)
    plt.plot(x, y, '+', c='r', label='Target')
    plt.xlim(x-size, x+size)
    plt.ylim(y-size, y+size)


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


def measure_fwhm(image, plot=True, printout=True):
    """
    Find the 2D FWHM of a background/continuum subtracted cutout image of a target.
    The target should be centered and cropped in the cutout.
    Use lcbg.utils.cutout for cropping targets.
    FWHM is estimated using the sigmas from a 2D gaussian fit of the target's flux.
    The FWHM is returned as a tuple of the FWHM in the x and y directions.

    Parameters
    ----------
    image : array like
        Input background/continuum subtracted cutout image.
    printout : bool
        Print out info.
    plot : bool
        To plot fit or not.

    Returns
    -------
    tuple : array of floats
        FWHM in x and y directions.
    """

    # Find FWHM
    # ----------

    fitted_line = fit_gaussian2d(image)

    # Find fitted center
    x_mean, y_mean = [i.value for i in [fitted_line.x_mean, fitted_line.y_mean]]

    # Estimate FWHM using gaussian_sigma_to_fwhm
    x_fwhm = fitted_line.x_stddev * gaussian_sigma_to_fwhm
    y_fwhm = fitted_line.y_stddev * gaussian_sigma_to_fwhm

    # Find half max
    hm = fitted_line(x_mean, y_mean) / 2.

    # Find the mean of the x and y direction
    mean_fwhm = np.mean([x_fwhm, y_fwhm])
    mean_fwhm = int(np.round(mean_fwhm))

    # Print info about fit and FWHM
    # ------------------------------

    if printout:
        print("Image Max: {}".format(image.max()))
        print("Amplitude: {}".format(fitted_line.amplitude.value))
        print("Center: ({}, {})".format(x_mean, y_mean))
        print("Sigma = ({}, {})".format(fitted_line.x_stddev.value,
                                        fitted_line.y_stddev.value, ))
        print("Mean FWHM: {} Pix ".format(mean_fwhm))
        print("FWHM: (x={}, y={}) Pix ".format(x_fwhm, y_fwhm))

    if plot:

        fig, [ax0, ax1, ax2, ax3] = plot_fit(image, fitted_line)

        # Make x and y grid to plot to
        y_arange, x_arange = np.mgrid[:image.shape[0], :image.shape[1]]

        # Plot input image with FWHM and center
        # -------------------------------------

        ax0.imshow(image, cmap='gray_r')

        ax0.axvline(x_mean - x_fwhm / 2, c='c', linestyle="--", label="X FWHM")
        ax0.axvline(x_mean + x_fwhm / 2, c='c', linestyle="--")

        ax0.axhline(y_mean - y_fwhm / 2, c='g', linestyle="--", label="Y FWHM")
        ax0.axhline(y_mean + y_fwhm / 2, c='g', linestyle="--")

        ax0.set_title("Center and FWHM Plot")
        ax0.legend()

        # Plot X fit
        # ----------

        ax2.axvline(x_mean, linestyle="-", label="Center")
        ax2.axvline(x_mean - x_fwhm / 2, c='c', linestyle="--", label="X FWHM")
        ax2.axvline(x_mean + x_fwhm / 2, c='c', linestyle="--")
        ax2.axhline(hm, c="black", linestyle="--", label="Half Max")

        ax2.legend()

        # Plot Y fit
        # ----------

        ax3.axvline(y_mean, linestyle="-", label="Center")
        ax3.axvline(y_mean - y_fwhm / 2, c='g', linestyle="--", label="Y FWHM")
        ax3.axvline(y_mean + y_fwhm / 2, c='g', linestyle="--")
        ax3.axhline(hm, c="black", linestyle="--", label="Half Max")

        ax3.legend()

        plt.show()

    return np.array([x_fwhm, y_fwhm])





