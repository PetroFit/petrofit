import numpy as np

from photutils.aperture import EllipticalAnnulus, EllipticalAperture

from .plotting import imshow, subplots, package_plot_style

__all__ = [
    "radial_elliptical_aperture",
    "radial_elliptical_annulus",
    "radial_photometry",
]


def radial_elliptical_aperture(position, r, elong=1.0, theta=0.0):
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


def radial_elliptical_annulus(position, r, dr, elong=1.0, theta=0.0):
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


def radial_photometry(
    image,
    position,
    r_list,
    error=None,
    mask=None,
    elong=1.0,
    theta=0.0,
    method='exact',
    plot=False,
    ax=None, 
    vmin=0,
    vmax=None,
    imshow_kwargs=None,
    aperture_plot_kwargs=None,
    subplots_kwargs={},
):
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

    plot : bool
        Plot the target and apertures.

    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If not provided, a new figure and axes will be created.

    vmin : int
        Min value for plot.

    vmax : int
        Max value for plot.

    imshow_kwargs : dict
        Additional keyword arguments to pass to the imshow function.

    aperture_plot_kwargs : dict
        Additional keyword arguments to pass to the aperture plot function.

    subplots_kwargs : dict
        Additional keyword arguments to pass to the subplots initialization.

    Returns
    -------
    photometry, aperture_area, error
        Returns photometry, aperture area (unmasked pixels) and error at each radius.
    """

    if imshow_kwargs is None:
        imshow_kwargs = {'cmap': 'viridis'}
    if aperture_plot_kwargs is None:
        aperture_plot_kwargs = {'color': 'white', 'alpha': 0.5}

    flux_arr = []
    error_arr = []
    area_arr = []

    if plot:
        with package_plot_style():
            if ax is None:
               fig, ax = subplots(1, 1, **subplots_kwargs)
            if vmax is not None:
                assert 'vmax' not in list(imshow_kwargs.keys()), "vmax and vmax in imshow_kwargs cannot be both set."
            if vmin is not None:
                assert 'vmin' not in list(imshow_kwargs.keys()), "vmin and vmin in imshow_kwargs cannot be both set."

            imshow_kwargs['vmax'] = image.mean() * 10 if vmax is None else vmax
            imshow_kwargs['vmin'] = vmin
            imshow(image, ax=ax, **imshow_kwargs)

            ax.set_title("Image and Aperture Radii")
            ax.set_xlabel("Pixels")
            ax.set_ylabel("Pixels")

    mask = ~mask if mask is not None else None
    for i, r in enumerate(r_list):
        aperture = radial_elliptical_aperture(position, r, elong=elong, theta=theta)

        photometric_value, photometric_err = aperture.do_photometry(
            data=image, error=error, mask=mask, method=method
        )
        aperture_area, aperture_area_err = aperture.do_photometry(
            data=np.ones_like(image), error=None, mask=mask, method=method
        )

        aperture_area = float(np.round(aperture_area, 6))
        photometric_value = float(np.round(photometric_value, 6))
        photometric_err = (
            float(np.round(photometric_err, 6)) if photometric_err.size > 0 else np.nan
        )

        if np.isnan(photometric_value):
            raise Exception("Nan photometric_value")

        if plot:
            with package_plot_style():
                aperture.plot(ax, **aperture_plot_kwargs)

        flux_arr.append(photometric_value)
        area_arr.append(aperture_area)
        error_arr.append(photometric_err)

    return np.array(flux_arr), np.array(area_arr), np.array(error_arr)
