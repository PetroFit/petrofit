import numpy as np

from matplotlib import pyplot as plt

from photutils.aperture import EllipticalAnnulus, EllipticalAperture

__all__ = [
    'plot_apertures', 'radial_elliptical_aperture',
    'radial_elliptical_annulus', 'radial_photometry', ]


def plot_apertures(image=None, apertures=[], vmin=None, vmax=None, color='white', lw=1.5):
    """
    Plot apertures on image

    Parameters
    ----------
    image : numpy.ndarray
        2D image array.

    apertures : list
        List of photutils Apertures.

    vmin, vmax : float
        vmax and vmin values for plot.

    color : string
        Matplotlib color for the apertures, default=White.

    lw : float
        Line width of aperture outline.
    """
    if image is not None:
        plt.imshow(image, cmap='Greys_r', vmin=vmin, vmax=vmax)

    for aperture in apertures:
        aperture.plot(axes=plt.gca(), color=color, lw=lw)

    if image is not None or apertures is not None or len(apertures) > 0:
        plt.title('Apertures')


def radial_elliptical_aperture(position, r, elong=1., theta=0.):
    """
    Helper function given a radius, elongation and theta,
    will make an elliptical aperture.

    Parameters
    ----------
    position : tuple
        (x, y) coords for center of aperture.

    r : int or float
        Semi-major radius of the aperture.

    elong : float
        Elongation.

    theta : float
        Orientation in rad.

    Returns
    -------
    EllipticalAperture
    """
    a, b = r, r / elong
    return EllipticalAperture(position, a, b, theta=theta)


def radial_elliptical_annulus(position, r, dr, elong=1., theta=0.):
    """
    Helper function given a radius, elongation and theta,
    will make an elliptical annulus.

    Parameters
    ----------
    position : tuple
        (x, y) coords for center of aperture

    r : int or float
        Semi-major radius of the inner ring

    dr : int or float
        Thickness of annulus (outer ring = r + dr).

    elong : float
        Elongation.

    theta : float
        Orientation in rad.

    Returns
    -------
    EllipticalAnnulus
    """

    a_in, b_in = r, r / elong
    a_out, b_out = r + dr, (r + dr) / elong

    return EllipticalAnnulus(position, a_in, a_out, b_out, theta=theta)


def radial_photometry(image, position, r_list, error=None, mask=None, elong=1., theta=0.,
                      plot=False, vmin=0, vmax=None, method='exact'):
    """
    Core photometry function.  Given a position, a list of radii and the shape
    of apertures, calculate the photometry of the target in the image.

    Parameters
    ----------
    image : 2D array
        Image to preform photometry on.

    position : tuple
        (x, y) position in pixels.

    r_list : list
        A list of radii for apertures.

    error : 2D array
        Error map of the image.

    mask : 2D array
        Boolean array with True meaning that pixel is unmasked.

    elong : float
        Elongation.

    theta : float
        Orientation in rad.

    plot : bool
        Plot the target and apertures.

    vmin : int
        Min value for plot.

    vmax : int
        Max value for plot.

    method : {'exact', 'center', 'subpixel'}, optional
            The method used to determine the overlap of the aperture on
            the pixel grid.  Not all options are available for all
            aperture types.  Note that the more precise methods are
            generally slower.  The following methods are available:

                * ``'exact'`` (default):
                  The the exact fractional overlap of the aperture and
                  each pixel is calculated.  The returned mask will
                  contain values between 0 and 1.

                * ``'center'``:
                  A pixel is considered to be entirely in or out of the
                  aperture depending on whether its center is in or out
                  of the aperture.  The returned mask will contain
                  values only of 0 (out) and 1 (in).

                * ``'subpixel'``
                  A pixel is divided into subpixels (see the
                  ``subpixels`` keyword), each of which are considered
                  to be entirely in or out of the aperture depending on
                  whether its center is in or out of the aperture.  If
                  ``subpixels=1``, this method is equivalent to
                  ``'center'``.  The returned mask will contain values
                  between 0 and 1.

    Returns
    -------
    photometry, aperture_area, error
        Returns photometry, aperture area (unmasked pixels) and error at each radius.
    """

    flux_arr = []
    error_arr = []
    area_arr = []

    if plot:
        ax = plt.gca()
        plt.imshow(image, vmin=vmin, vmax=image.mean() * 10 if vmax is None else vmax)
        ax.set_title("Image and Aperture Radii")
        ax.set_xlabel("Pixels")
        ax.set_ylabel("Pixels")

    mask = ~mask if mask is not None else None
    for i, r in enumerate(r_list):
        aperture = radial_elliptical_aperture(position, r, elong=elong, theta=theta)

        photometric_value, photometric_err = aperture.do_photometry(data=image, error=error, mask=mask, method=method)
        aperture_area, aperture_area_err = aperture.do_photometry(data=np.ones_like(image), error=None,
                                                                  mask=mask, method=method)

        aperture_area = float(np.round(aperture_area, 6))
        photometric_value = float(np.round(photometric_value, 6))
        photometric_err = float(np.round(photometric_err, 6)) if photometric_err.size > 0 else np.nan

        if np.isnan(photometric_value):
            raise Exception("Nan photometric_value")

        if plot:
            aperture.plot(plt.gca(), color='w', alpha=0.5)

        flux_arr.append(photometric_value)
        area_arr.append(aperture_area)
        error_arr.append(photometric_err)

    return np.array(flux_arr), np.array(area_arr), np.array(error_arr)
