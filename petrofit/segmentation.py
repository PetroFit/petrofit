from copy import deepcopy

import numpy as np

from photutils import detect_threshold
from photutils import deblend_sources
from photutils import detect_sources
from photutils.segmentation.properties import SourceProperties

from astropy.convolution import Gaussian2DKernel
from astropy.stats import gaussian_fwhm_to_sigma

from matplotlib import pyplot as plt

__all__ = [
    'plot_segments', 'plot_segment_residual', 'get_source_position', 'get_source_elong',
    'get_source_ellip', 'get_source_theta', 'make_kernel', 'segm_mask', 'masked_segm_image',
    'make_segments', 'deblend_segments'
]

def plot_segments(segm, image=None, vmin=None, vmax=None, alpha=0.5):
    """
    Plot segmented areas over an image (if provided)
    """

    cmap = segm.make_cmap(random_state=np.random.randint(1000000))

    if image is not None:
        plt.imshow(image, vmin=vmin, vmax=vmax, cmap="gist_gray")

    plt.imshow(segm, cmap=cmap, alpha=alpha)

    plt.title('Segmentation Image')


def plot_segment_residual(segm, image, vmin=None, vmax=None):
    """
    Plot segment subtracted image (residual)
    """
    temp = image.copy()
    temp[np.where(segm.data != 0)] = 0.0
    plt.imshow(temp, vmin=vmin, vmax=vmax)


def get_source_position(obj):
    """Return max x, y value of a SourceProperties or catalog row"""
    if isinstance(obj, SourceProperties):
        x, y = obj.maxval_xpos.value, obj.maxval_ypos.value
    else:
        x, y = obj['maxval_xpos'], obj['maxval_ypos']
    return x, y


def get_source_elong(obj):
    """ Return SourceProperties elongation"""
    return obj.elongation.value if isinstance(obj, SourceProperties) else obj['elongation']


def get_source_ellip(obj):
    """ Return SourceProperties ellipticity"""
    return obj.ellipticity.value if isinstance(obj, SourceProperties) else obj['ellipticity']


def get_source_theta(obj):
    """ Return SourceProperties orientation in rad"""
    return obj.orientation.to('rad').value if isinstance(obj, SourceProperties) else np.deg2rad(obj['orientation'])


def make_kernel(fwhm, kernel_size):
    sigma = fwhm * gaussian_fwhm_to_sigma
    kernel = Gaussian2DKernel(sigma, x_size=kernel_size, y_size=kernel_size)
    kernel.normalize()
    return kernel


def segm_mask(obj, segm, mask_background=False):
    """
    Given a segmentation and a target with an ID, returns a mask
    with all other sources masked in the original image.

    Parameters
    ----------
    obj : int or photutils.segmentation.properties.SourceProperties
        The catalog object for the target or id of target in the segmentation object
    segm : photutils.segmentation.core.SegmentationImage
        Segmentation image
    mask_background : bool
        Option to also mask out all the un-segmented background pixels.

    Returns
    -------
    mask : bool array
    """

    if isinstance(obj, SourceProperties):
        obj = obj.id

    mask = (segm.data == obj)
    if not mask_background:
        mask = ((segm.data == 0) | mask)
    return mask


def masked_segm_image(obj, image, segm, fill=None, mask_background=False):
    """
    Returns a masked image of the original image by masking out other sources
    Parameters
    ----------
    obj : int or photutils.segmentation.properties.SourceProperties
        The catalog object for the target or id of target in the segmentation object
    image : CCDData or array
        Image to mask
    segm : photutils.segmentation.core.SegmentationImage
        Segmentation image
    fill : float
        Fill in the masked pixels with this value
    mask_background : bool
        Option to also mask out all the un-segmented background pixels.

    Returns
    -------
    masked_image : CCDData or array
    """

    fill = np.nan if fill is None else fill
    mask = segm_mask(obj, segm, mask_background)

    masked_image = deepcopy(image)

    if isinstance(masked_image, np.ndarray):
        masked_image[np.invert(mask)] = fill
    else:
        masked_image.data[np.invert(mask)] = fill

    return masked_image


def make_segments(image, npixels=None, nsigma=3., fwhm=8., kernel_size=4):
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
    nsigma : float or image array
        The number of standard deviations per pixel above the
        ``background`` for which to consider a pixel as possibly being
        part of a source.
    fwhm : float
        FWHM of smoothing gaussian kernel
    kernel_size : int
        Size of smoothing kernel

    Returns
    -------
    segment_image : `~photutils.segmentation.SegmentationImage` or `None`
        A 2D segmentation image, with the same shape as ``data``, where
        sources are marked by different positive integer values.  A
        value of zero is reserved for the background.  If no sources
        are found then `None` is returned.
    """

    if npixels is None:
        npixels = fwhm ** 2

    kernel = make_kernel(fwhm, kernel_size) if kernel_size else None

    # Make detection threshold
    if isinstance(nsigma, int):
        threshold = detect_threshold(image, nsigma=nsigma)
    else:
        threshold = nsigma

    return detect_sources(image, threshold, npixels=npixels, filter_kernel=kernel, )


def deblend_segments(image, segm, npixels=None, fwhm=8., kernel_size=4, nlevels=30, contrast=1/1000):
    """
    Deblend overlapping sources labeled in a segmentation image.

    Parameters
    ----------
    image : array like
        Input image
    segm : `~photutils.segmentation.SegmentationImage` or `None`
        A 2D segmentation image, with the same shape as ``data``, where
        sources are marked by different positive integer values.  A
        value of zero is reserved for the background.  If no sources
        are found then `None` is returned.
    npixels : int
        The number of connected pixels, each greater than ``threshold``,
        that an object must have to be detected.  ``npixels`` must be a
        positive integer.
    fwhm : float
        FWHM of smoothing gaussian kernel
    kernel_size : int
        Size of smoothing kernel
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

    if npixels is None:
        npixels = fwhm ** 2

    kernel = make_kernel(fwhm, kernel_size) if kernel_size else None

    segm_deblend = deblend_sources(image, segm,
                                   npixels=npixels, filter_kernel=kernel,
                                   nlevels=nlevels, contrast=contrast)

    return segm_deblend