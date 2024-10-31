import re

import numpy as np

from scipy.interpolate import interp1d

from astropy.wcs.utils import proj_plane_pixel_scales
from astropy import units as u
from astropy.coordinates import SkyCoord

from matplotlib import pyplot as plt


__all__ = [
    "match_catalogs",
    "angular_to_pixel",
    "pixel_to_angular",
    "elliptical_area_to_r",
    "circle_area_to_r",
    "get_interpolated_values",
    "closest_value_index",
    "plot_target",
    "cutout_subtract",
    "hst_flux_to_abmag",
    "make_radius_list",
    "natural_sort",
    "mpl_tick_frame",
    "ellip_to_elong",
    "elong_to_ellip",
]


def natural_sort(l):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split("([0-9]+)", key)]
    return sorted(l, key=alphanum_key)


def ellip_to_elong(ellip):
    return 1 / (1 - ellip)


def elong_to_ellip(elong):
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

    if not hasattr(pixel_size, "unit"):
        pixel_size = pixel_size * u.pix

    angular_diameter = pixel_size * pixel_scale.to(u.arcsec / u.pix)
    return angular_diameter


def elliptical_area_to_r(area, elong):
    a = np.sqrt(elong * area / (np.pi))
    b = a / elong
    return a, b


def circle_area_to_r(area):
    return np.sqrt(area / np.pi)


def get_interpolated_values(x, y, num=5000, kind="cubic"):
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


def plot_target(
    image, position, size=None, c="r", lw=None, vmin=None, vmax=None, marker_base_size=2
):
    """
    Plot an image with a target marker.

    Parameters
    ----------
    image : np.ndarray or object with `data` attribute
        The image to be displayed.
    position : tuple of int
        (x, y) coordinates of the target location.
    size : int, optional
        The pixel size around the target to display.
        If not specified, it defaults to the maximum dimension of the image.
    c : str, optional
        Color of the target marker. Default is red (`'r'`).
    lw : int or float, optional
        Line width of the target marker.
    vmin, vmax : int or float, optional
        Values to anchor the colormap.
    marker_base_size : int, optional
        Base size of the marker which gets scaled relative to the image size.
        Default is 2.
    Notes
    -----
    The target is plotted as a red '+' at the given position. The displayed
    region is determined by the `size` parameter centered at the target position.
    """

    if size is None:
        size = max(image.shape)
    x, y = position

    # Calculate marker size relative to the average size of the image dimensions
    marker_size = np.mean(image.shape) / 20 * marker_base_size
    plt.imshow(image, vmin=vmin, vmax=vmax)
    plt.plot(x, y, "+", c=c, label="Target", markersize=marker_size, markeredgewidth=lw)
    plt.xlim(x - (size / 2.0), x + (size / 2.0))
    plt.ylim(y - (size / 2.0), y + (size / 2.0))


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


def mpl_tick_frame(ax=None, minorticks=True, tick_fontsize=None):
    if ax is None:
        ax = plt.gca()
    if minorticks:
        ax.minorticks_on()
    ax.tick_params(
        which="minor",
        direction="in",
        top=True,
        right=True,
        width=1.5,
        length=8 / 2,
        labelsize=tick_fontsize,
    )
    ax.tick_params(
        which="major",
        direction="in",
        top=True,
        right=True,
        width=1.5,
        length=8,
        labelsize=tick_fontsize,
    )
