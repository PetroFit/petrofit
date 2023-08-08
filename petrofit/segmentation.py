from copy import deepcopy

import numpy as np

from astropy.nddata import CCDData, Cutout2D

from photutils import deblend_sources
from photutils import detect_sources
from photutils.segmentation import SourceCatalog
from photutils.isophote import EllipseGeometry, Ellipse

from matplotlib import pyplot as plt

__all__ = [
    'plot_segments', 'plot_segment_residual', 'get_source_position', 'get_source_elong',
    'get_source_ellip', 'get_source_theta', 'get_amplitude_at_r', 'order_cat', 'segm_mask',
    'masked_segm_image', 'make_segments', 'deblend_segments', 'make_catalog'
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
        range :math:`0 < PA <= \pi`. Avoid using as starting position
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

    if isinstance(source, int) or isinstance(source, SourceCatalog):
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
                 plot=True, vmax=None, vmin=None):
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
            fig, ax = plt.subplots(1, 2)
            plt.sca(ax[0])
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
