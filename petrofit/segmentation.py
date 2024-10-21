import warnings
from copy import deepcopy

import numpy as np

from astropy.nddata import CCDData, Cutout2D
from astropy.stats import sigma_clipped_stats, sigma_clip
from astropy.utils.exceptions import AstropyWarning

from photutils.segmentation import SourceCatalog, deblend_sources, detect_sources
from photutils.isophote import EllipseGeometry, Ellipse

from matplotlib import pyplot as plt

from .modeling.fitting import fit_background, model_to_image
from .photometry import radial_photometry
from .utils import mpl_tick_frame

__all__ = [
    'plot_segments', 'plot_segment_residual', 'get_source_position', 'get_source_elong',
    'get_source_ellip', 'get_source_theta', 'get_amplitude_at_r', 'order_cat', 'segm_mask',
    'masked_segm_image', 'make_segments', 'deblend_segments', 'make_catalog', 'source_photometry'
]


def plot_segments(segm, image=None, vmin=None, vmax=None, alpha=0.5, title=None):
    """
    Plot segmented areas over an image (2D array, if provided)
    """

    cmap = segm.make_cmap(seed=np.random.randint(1000000))

    if image is not None:
        plt.imshow(image, vmin=vmin, vmax=vmax, cmap="gist_gray")

    plt.imshow(segm, cmap=cmap, alpha=alpha)

    if title is not None:
        plt.title(title)

    plt.xlabel("Pixels")
    plt.ylabel("Pixels")


def plot_segment_residual(segm, image, vmin=None, vmax=None):
    """
    Plot segment subtracted image (residual)
    """
    temp = image.copy()
    temp[np.where(segm.data != 0)] = 0.0
    plt.imshow(temp, vmin=vmin, vmax=vmax)


def get_source_position(source):
    """Return max x, y value of a SourceCatalog or catalog row"""
    if isinstance(source, SourceCatalog):
        x, y = source.maxval_xindex, source.maxval_yindex
    else:
        x, y = source['maxval_xindex'], source['maxval_yindex']
    return x, y


def get_source_elong(source):
    """ Return SourceCatalog elongation"""
    return source.elongation.value if isinstance(source, SourceCatalog) else source['elongation']


def get_source_ellip(source):
    """ Return SourceCatalog ellipticity"""
    return source.ellipticity.value if isinstance(source, SourceCatalog) else source['ellipticity']


def get_source_theta(source):
    """ Return SourceCatalog orientation in rad"""
    return source.orientation.to('rad').value if isinstance(source, SourceCatalog) else np.deg2rad(source['orientation'])


def get_amplitude_at_r(r, image, x0, y0, ellip, theta):
    """
    Finds the amplitude at an isophotal radius `r`.

    Parameters
    ----------

    r : float or int
        Isophotal radius in pixels.

    image : CCDData or array
        Image to of the source.

    x0, y0 : float
        The center pixel coordinate of the ellipse.

    ellip : ellipticity
        The ellipticity of the ellipse.

    theta : float
        The position angle (in radians) of the semimajor axis in
        relation to the positive x axis of the image array (rotating
        towards the positive y axis). Position angles are defined in the
        range :math:`0 < PA <= \\pi`. Avoid using as starting position
        angle of 0., since the fit algorithm may not work properly.
        When the ellipses are such that position angles are near either
        extreme of the range, noise can make the solution jump back and
        forth between successive isophotes, by amounts close to 180
        degrees.

    Returns
    -------

    amplitude_at_r : float or np.nan

    """

    if isinstance(image, CCDData) or isinstance(image, Cutout2D):
        image = image.data

    r = float(r)

    try:
        # Define EllipseGeometry using ellip and theta
        g = EllipseGeometry(x0, y0, 1., ellip, theta)

        # Create Ellipse model
        ellipse = Ellipse(image, geometry=g)

        # Fit isophote at r_eff
        iso = ellipse.fit_isophote(r)

        # Get flux at r_eff
        amplitude = iso.intens

    except Exception as exception:
        import warnings
        warnings.warn("Amplitude could not be computed, returning np.nan. Exception: {}".format(str(exception)), Warning)
        amplitude = np.nan

    return amplitude


def order_cat(cat, key='area', reverse=True):
    """
    Sort a catalog by largest area and return the argsort

    Parameters
    ----------
    cat : `SourceCatalog` instance
        A `SourceCatalog` instance containing the properties of each source.

    key : string
        Key to sort.

    reverse : bool
        Reverse sorting order. Default is `True` to place largest values on top.

    Returns
    -------
    output : list
        A list of catalog indices ordered by largest area.
    """
    table = cat.to_table()[key]
    order_all = table.argsort()
    if reverse:
        return list(reversed(order_all))
    return order_all


def segm_mask(source, segm, mask_background=False):
    """
    Given a segmentation and a target with an label, returns a mask
    with all other sources masked in the original image.

    Parameters
    ----------

    source : list or int or photutils.segmentation.properties.SourceCatalog
        The catalog object for the target or label of target in the segmentation object.

    segm : photutils.segmentation.core.SegmentationImage
        Segmentation image.

    mask_background : bool
        Option to also mask out all the un-segmented background pixels.

    Returns
    -------

    mask : bool array
    """

    if isinstance(source, int) or isinstance(source, np.integer) or isinstance(source, SourceCatalog):
        sources = [source]
    elif isinstance(source, list):
        sources = source
    else:
        raise TypeError('Input should be a label (int or photutils SourceCatalog) or a list of such items')

    if not mask_background:
        sources.append(0)

    mask = None
    for source in sources:
        if isinstance(source, SourceCatalog):
            source = [source.label]

        if mask is None:
            mask = (segm.data == source)
        else:
            mask = ((segm.data == source) | mask)
    return mask


def masked_segm_image(source, image, segm, fill=None, mask_background=False):
    """
    Returns a masked image of the original image by masking out other sources

    Parameters
    ----------

    source : list or int or photutils.segmentation.properties.SourceCatalog
        The catalog object for the target or label of target in the segmentation object.

    image : CCDData or array
        Image to mask.

    segm : photutils.segmentation.core.SegmentationImage
        Segmentation image.

    fill : float
        Fill in the masked pixels with this value.

    mask_background : bool
        Option to also mask out all the un-segmented background pixels.

    Returns
    -------

    masked_image : CCDData or array
    """

    fill = np.nan if fill is None else fill
    mask = segm_mask(source, segm, mask_background)

    masked_image = deepcopy(image)

    if isinstance(masked_image, np.ndarray):
        masked_image[np.invert(mask)] = fill
    else:
        masked_image.data[np.invert(mask)] = fill

    return masked_image


def make_segments(image, npixels=None, threshold=3.):
    """
    Segment an image.

    Parameters
    ----------
    image : array like
        Input image

    npixels : int
        The number of connected pixels, each greater than ``threshold``,
        that an object must have to be detected.  ``npixels`` must be a
        positive integer.

    threshold : float or array-like
        The data value or pixel-wise data values to be used for the
        detection threshold. A 2D ``threshold`` must have the same shape
        as ``data``. See `~photutils.segmentation.detect_threshold` for
        one way to create a ``threshold`` image.

    Returns
    -------

    segment_image : `~photutils.segmentation.SegmentationImage` or `None`
        A 2D segmentation image, with the same shape as ``data``, where
        sources are marked by different positive integer values.  A
        value of zero is reserved for the background.  If no sources
        are found then `None` is returned.
    """
    return detect_sources(image, threshold, npixels=npixels)


def deblend_segments(image, segm, npixels=None, nlevels=30, contrast=1/1000):
    """
    Deblend overlapping sources labeled in a segmentation image.

    Parameters
    ----------
    image : array like
        Input image.

    segm : `~photutils.segmentation.SegmentationImage` or `None`
        A 2D segmentation image, with the same shape as ``data``, where
        sources are marked by different positive integer values.  A
        value of zero is reserved for the background.  If no sources
        are found then `None` is returned.

    npixels : int
        The number of connected pixels, each greater than ``threshold``,
        that an object must have to be detected.  ``npixels`` must be a
        positive integer.

    nlevels : int, optional
        The number of multi-thresholding levels to use.  Each source
        will be re-thresholded at ``nlevels`` levels spaced
        exponentially or linearly (see the ``mode`` keyword) between its
        minimum and maximum values within the source segment.

    contrast : float, optional
        The fraction of the total (blended) source flux that a local
        peak must have (at any one of the multi-thresholds) to be
        considered as a separate object.  ``contrast`` must be between 0
        and 1, inclusive.  If ``contrast = 0`` then every local peak
        will be made a separate object (maximum deblending).  If
        ``contrast = 1`` then no deblending will occur.  The default is
        0.001, which will deblend sources with a 7.5 magnitude
        difference.

    Returns
    -------
    segment_image : `~photutils.segmentation.SegmentationImage`
        A segmentation image, with the same shape as ``data``, where
        sources are marked by different positive integer values.  A
        value of zero is reserved for the background.
    """
    segm_deblend = deblend_sources(image, segm, npixels=npixels,
                                   nlevels=nlevels, contrast=contrast)

    return segm_deblend


def make_catalog(image, threshold, wcs=None, deblend=True,
                 npixels=None, nlevels=30, contrast=1/1000,
                 plot=True, vmax=None, vmin=None, figsize=None):
    """
    This function constructs a catalog using `PhotUtils`. The `petrofit.segmentation.make_segments` and
    `petrofit.segmentation.deblend_segments` functions are used to construct segmentation maps and
    the resulting segmentation is turned into a vatalog (`photutils.segmentation.catalog.SourceCatalog`)

    Parameters
    ----------
    image : array like
        Input image.

    threshold : float or array-like
        The data value or pixel-wise data values to be used for the
        detection threshold. A 2D ``threshold`` must have the same shape
        as ``data``. See `~photutils.segmentation.detect_threshold` for
        one way to create a ``threshold`` image.

    wcs : astropy.wcs.WCS
        Astropy WCS for catalog to convert source locations to RA-Dec.

    deblend : bool
        Flag to enable deblending of sources in the segmentation map.

    npixels : int
        The number of connected pixels, each greater than ``threshold``,
        that an object must have to be detected.  ``npixels`` must be a
        positive integer.

    nlevels : int, optional
        The number of multi-thresholding levels to use.  Each source
        will be re-thresholded at ``nlevels`` levels spaced
        exponentially or linearly (see the ``mode`` keyword) between its
        minimum and maximum values within the source segment.

    contrast : float, optional
        The fraction of the total (blended) source flux that a local
        peak must have (at any one of the multi-thresholds) to be
        considered as a separate object.  ``contrast`` must be between 0
        and 1, inclusive.  If ``contrast = 0`` then every local peak
        will be made a separate object (maximum deblending).  If
        ``contrast = 1`` then no deblending will occur.  The default is
        0.001, which will deblend sources with a 7.5 magnitude
        difference.

    plot : bool
        Flag to toggle plotting segmentation maps.

    vmin, vmax : float
        vmax and vmin values for plot.

    figsize : tuple
        Figure size.

    Returns
    -------
    cat, segm, segm_deblend
        cat : `photutils.segmentation.catalog.SourceCatalog`
            A catalog of sources

        segm : `photutils.segmentation.core.SegmentationImage`
            Segmentation map.

        segm_deblend : `photutils.segmentation.core.SegmentationImage`
            Deblended segmentation map.
    """

    if isinstance(image, CCDData) or isinstance(image, Cutout2D):
        image = image.data

    # Make segmentation map
    segm = make_segments(image,
                         threshold=threshold,
                         npixels=npixels)

    if plot and segm:
        # Make plots
        if deblend:
            fig, ax = plt.subplots(1, 2, figsize=figsize)
            plt.sca(ax[0])
        else:
            fig, ax = plt.subplots(1, 1, figsize=figsize)
        plot_segments(segm, image=image, vmax=vmax, vmin=vmin, title='Segmentation Map')

    # Deblend segmentation map
    segm_deblend = None
    if deblend:
        segm_deblend = deblend_segments(image, segm,
                                        contrast=contrast,
                                        nlevels=nlevels,
                                        npixels=npixels)

        if plot and segm_deblend:
            # Make plots
            plt.sca(ax[1])
            plot_segments(segm_deblend, image=image, vmax=vmax, vmin=vmin, title='Deblended Segmentation Map')

    # Make catalog
    cat = SourceCatalog(image, segm_deblend if deblend else segm, wcs=wcs)

    return cat, segm, segm_deblend


def source_photometry(source, image, segm_deblend, r_list, error=None, cutout_size=None,
                      bg_sub=False, sigma=3.0, sigma_type='clip', method='exact', mask_background=False,
                      plot=False, vmin=0, vmax=None, figsize=[12, 6]):
    """
    Aperture photometry on a PhotUtils `SourceProperties`.

    Parameters
    ----------
    source : `photutils.segmentation.SourceProperties`
        `SourceProperties` (an entry in a `SourceCatalog`)

    image : 2D array
        Image to preform photometry on.

    segm_deblend : `SegmentationImage`
        Segmentation map of the image.

    r_list : list
        List of aperture radii.

    error : 2D array
        Error image (optional).

    cutout_size : int
        Size of cutout.

    bg_sub : bool
        If the code should subtract the background using the `sigma` provided.

    sigma : float
        The sigma value used to determine noise pixels. Once the pixels above this value are masked,
        a 2D plane is fit to determine the background. The 2D plane model is then converted into an image and
        subtracted from the cutout of the target source. see the `sigma_type` on how this value will be used.

    sigma_type : {'clip', 'bound'}, optional
        The meaning of the provided sigma.
            * ``'clip'`` (default):
                Uses `astropy.stats.sigma_clipping.sigma_clip` to clip at the provided `sigma` std value.
                Note that `sigma` in this case is the number of stds above the mean.

            * ``'bound'``:
                After computing the mean of the image, clip at `mean - sigma` and `mean + sigma`.
                Note that `sigma` in this case is a value and not the number of stds above the mean.


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

    mask_background : bool
        Should background pixels, that are not part of any source in the segmentation map, be included?
        If False, only pixels inside the source's segmentation are unmasked.

    plot : bool
        Show plot of cutout and apertures.

    vmin : int
        Min value for plot.

    vmax : int
        Max value for plot.

    figsize : tuple
        Figure size.

    Returns
    -------

    flux_arr, area_arr, error_arr : (numpy.array, numpy.array, numpy.array)
        Tuple of arrays:

            * `flux_arr`: Photometric sum in aperture.

            * `area_arr`: Exact area of aperture.

            * `error_arr`: if error map is provided, error of measurements.
    """

    # Get source geometry
    # -------------------
    position = get_source_position(source)
    elong = get_source_elong(source)
    theta = get_source_theta(source)

    if cutout_size is None:
        cutout_size = np.ceil(max(r_list) * 3)

    cutout_size = int(cutout_size)
    if cutout_size % 2 == 1:
        cutout_size += 1

    # Error cutout
    # ------------
    masked_err = None
    if error is not None:
        masked_err = Cutout2D(error, position, cutout_size, mode='partial', fill_value=np.nan).data

    # Image Cutout
    # ------------
    full_masked_image = masked_segm_image(source, image, segm_deblend, fill=np.nan, mask_background=mask_background).data
    masked_nan_image = Cutout2D(full_masked_image, position, cutout_size, mode='partial', fill_value=np.nan)
    masked_image = masked_nan_image.data

    # Cutout for Stats
    # ----------------
    # This cutout has all sources masked
    stats_cutout_size = cutout_size  # max(source.bbox.ixmax - source.bbox.ixmin, source.bbox.iymax - source.bbox.iymin) * 2
    full_bg_image = masked_segm_image(0, image, segm_deblend, fill=np.nan, mask_background=False).data
    masked_stats_image = Cutout2D(full_bg_image, position, stats_cutout_size, mode='partial', fill_value=np.nan).data

    # Subtract Mean Plane
    # -------------------
    if bg_sub:
        if len(np.where(~np.isnan(masked_stats_image))[0]) > 10:
            with warnings.catch_warnings():

                warnings.simplefilter('ignore', AstropyWarning)
                if sigma_type.lower() == 'clip':
                    fit_bg_image = masked_stats_image
                    fit_bg_image = sigma_clip(fit_bg_image, sigma)

                elif sigma_type.lower() == 'bound':
                    mean, median, std = sigma_clipped_stats(masked_stats_image, sigma=3,
                                                            mask=np.isnan(masked_stats_image.data))

                    fit_bg_image = masked_stats_image
                    fit_bg_image[np.where(fit_bg_image > mean + std * sigma)] = np.nan
                    fit_bg_image[np.where(fit_bg_image < mean - std * sigma)] = np.nan
                else:
                    raise ("background image masking sigma type not understood, try 'clip' or 'bound'")

                fitted_model, _ = fit_background(fit_bg_image, sigma=None)

                masked_image -= model_to_image(fitted_model, cutout_size)
                if sigma_type.lower() == 'bound':
                    masked_image = np.clip(masked_image, - sigma, np.inf)

        elif plot:
            print("bg_sub: Not enough datapoints, did not subtract.")

    # Make mask
    # ---------
    mask = np.ones_like(masked_image)
    mask[np.where(np.isnan(masked_image))] = 0
    mask = mask.astype(bool)

    position = np.array(masked_image.data.shape) / 2.

    if plot:
        print(source.label)
        fig, ax = plt.subplots(1, 2, figsize=figsize)

    if plot:
        plt.sca(ax[0])

    flux_arr, area_arr, error_arr = radial_photometry(masked_image, position, r_list, error=masked_err, mask=mask,
                                                      elong=elong, theta=theta, plot=plot, vmin=vmin, vmax=vmax,
                                                      method=method)

    if plot:
        plt.sca(ax[1])
        plt.plot(r_list, flux_arr, c='tab:blue', linewidth=3, zorder=3)
        for r in r_list:
            plt.axvline(r, alpha=0.5, c='gray')
        plt.title("Curve of Growth")
        plt.xlabel("Radius in Pixels")
        plt.ylabel("Flux Enclosed")
        mpl_tick_frame(ax=ax[1], minorticks=True)
        plt.show()

        r = max(r_list)
        fig, ax = plt.subplots(1, 1, figsize=figsize)

        plt.plot(masked_image[int(position[1]), :], c='tab:blue', linewidth=3, zorder=3)
        plt.axhline(0, c='black')
        # plt.axhline(noise_sigma, c='b')
        plt.axvline(position[0], linestyle='--', c='gray')
        plt.axvline(position[0] + r, alpha=0.5, c='gray')
        plt.axvline(position[0] - r, alpha=0.5, c='gray')
        plt.xlabel("Slice Along X [pix]")
        plt.ylabel("Pixel Value")
        mpl_tick_frame(ax=ax, minorticks=True)

        fig, ax = plt.subplots(1, 1, figsize=figsize)
        plt.plot(masked_image[:, int(position[0])], c='tab:blue', linewidth=3, zorder=3)
        plt.axhline(0, c='black')
        # plt.axhline(noise_sigma, c='b')
        plt.axvline(position[0], linestyle='--', c='gray')
        plt.axvline(position[0] + r, alpha=0.5, c='gray')
        plt.axvline(position[0] - r, alpha=0.5, c='gray')
        plt.xlabel("Slice Along Y [pix]")
        plt.ylabel("Pixel Value")
        mpl_tick_frame(ax=ax, minorticks=True)

    return flux_arr, area_arr, error_arr
