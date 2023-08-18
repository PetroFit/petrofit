import numpy as np

from astropy.modeling import models, fitting
from astropy.modeling.optimizers import DEFAULT_ACC, DEFAULT_EPS
from astropy.stats import sigma_clip, gaussian_sigma_to_fwhm
from astropy.nddata import CCDData, Cutout2D
from astropy.convolution.utils import discretize_model

from matplotlib import pyplot as plt

__all__ = [
    'fit_model', 'model_to_image', 'fit_background',
    'fit_gaussian2d', 'print_model_params', 'plot_fit', 'measure_fwhm'
]


def fit_model(image, model, weights=None, maxiter=5000,
              epsilon=DEFAULT_EPS, acc=DEFAULT_ACC, estimate_jacobian=False):
    """
    Wrapper function to conveniently fit an image to an input model.

    Parameters
    ----------
    image : array
        2D array to fit.

    model : `~astropy.modeling.FittableModel`
        Astropy model to sample from. The model must be 2D and the inputs
        are (x, y) pixel coordinates.

    weights : array
        Weights for fitting.
        For data with Gaussian uncertainties, the weights should be
        1/sigma.

    maxiter : int
        maximum number of iterations

    epsilon : float
        A suitable step length for the forward-difference
        approximation of the Jacobian (if model.fjac=None). If
        epsfcn is less than the machine precision, it is
        assumed that the relative errors in the functions are
        of the order of the machine precision.

    acc : float
        Relative error desired in the approximate solution

    estimate_jacobian : bool
        If False (default) and if the model has a fit_deriv method,
        it will be used. Otherwise the Jacobian will be estimated.
        If True, the Jacobian will be estimated in any case.

    Returns
    -------
    fitted_model, fitter

        * fitted_model : `~astropy.modeling.FittableModel`
            A copy of the input model with parameters set by the fitter.

        * fitter : LevMarLSQFitter
            Fitter used to estimate and set model parameters.
    """

    # Make x and y grid to fit to
    y_arange, x_arange = np.where(~(np.isnan(image)))

    z = image[(y_arange, x_arange)]

    # Prepare weights
    w = None
    if weights is not None:
        assert weights.shape == image.shape, "RMS array does not have the same shape as input image."
        w = weights[(y_arange, x_arange)]

    # Fit model to grid
    fitter = fitting.LevMarLSQFitter()
    fitted_model = fitter(model, x_arange, y_arange, z, weights=w,
                          maxiter=maxiter, epsilon=epsilon, acc=acc, estimate_jacobian=estimate_jacobian)

    return fitted_model, fitter


def _validate_image_size(size):
    """
    Helper function to validate image size.
    Input size (pixels) should be an integers or a tuple of two integers.

    Parameters
    ----------
    size : int or tuple
        User size input

    Returns
    -------
    (x_size, y_size) : tuple
        Validated size as tuple.
    """

    error_message = "Input size (pixels) should be an integers or a tuple of two integers."

    if type(size) in [list, tuple, np.array]:
        assert len(size) == 2, error_message
        x_size, y_size = size

    elif np.issubdtype(type(size), np.number):
        x_size = size
        y_size = size

    else:
        raise ValueError(error_message)

    assert not x_size % 1 and not y_size % 1, error_message

    x_size = int(x_size)
    y_size = int(y_size)

    return x_size, y_size


def model_center_to_image_origin(center, size):
    """
    Given the center and size of an image, find the origin coordinate of the image.
    """

    x_size, y_size = _validate_image_size(size)

    center_error_message = "Center should be a tuple of two integers."
    assert type(center) in [list, tuple, np.array], center_error_message
    assert len(center) == 2, center_error_message

    origin = np.array(center) - np.floor_divide(np.array([x_size, y_size]), 2)

    return tuple(origin)


def model_to_image(model, size, mode='center', factor=1, center=None):
    """
    Converts 2D models into images using `astropy.convolution.utils.discretize_model`.

    Parameters
    ----------
    model : `~astropy.modeling.FittableModel` or callable.
        Analytic model function to be discretized. Callables, which are not an
        instances of `~astropy.modeling.FittableModel` are passed to
        `~astropy.modeling.custom_model` and then evaluated.

    size : int or tuple
        The x and y size (in pixels) of the image in pixels (must be an whole number).
        If only a single integer is provided, an image of equal x and y size
        is generated. If tuple is provided (x_size, y_size) is assumed (N.B reverse of `numpy.array.shape` output).

    mode : str, optional
        One of the following modes (`astropy.convolution.utils.discretize_model`):
            * ``'center'`` (default)
                Discretize model by taking the value
                at the center of the bin.
            * ``'linear_interp'``
                Discretize model by linearly interpolating
                between the values at the corners of the bin.
                For 2D models interpolation is bilinear.
            * ``'oversample'``
                Discretize model by taking the average
                on an oversampled grid.
            * ``'integrate'``
                Discretize model by integrating the model
                over the bin using `scipy.integrate.quad`.
                Very slow.

    factor : float or int
        Factor of oversampling. Default = 1 (no oversampling).

    center : tuple
        (x, y) Coordinate of the center of the image (in pixels).
        The origin of the image is defined as `origin = center - floor_divide(size, 2)`
        (i.e the image will range from (origin -> origin + size)). If None, the origin
        of the image is assumed to be at (0, 0) (i.e `center = floor_divide(size, 2)`).

    Returns
    -------
    array : `numpy.array`
        Model image
    """

    x_size, y_size = _validate_image_size(size)

    if center is None:
        x_origin, y_origin = (0, 0)
    else:
        x_origin, y_origin = model_center_to_image_origin(center, size)

    return discretize_model(
        model=model,
        x_range=[x_origin, x_origin + x_size],
        y_range=[y_origin, y_origin + y_size],
        mode=mode,
        factor=factor)


def fit_background(image, model=models.Planar2D(), sigma=3.0):
    """
    Fit sigma clipped background image using a user provided model.

    Parameters
    ----------
    image : array
        2D array to fit.

    model : `~astropy.modeling.FittableModel`
        AstroPy model to sample from. `Planar2D` is used by default.

    sigma : float or None
        The sigma value used to determine noise pixels. Once the pixels above this value are masked,
        the model provided is fit to determine the background.

    Returns
    -------
    fitted_model, fitter

        * fitted_model : `~astropy.modeling.FittableModel`
            A copy of the input model with parameters set by the fitter.

        * fitter : LevMarLSQFitter
            Fitter used to estimate and set model parameters.
    """
    fit_bg_image = image
    if sigma is not None:
        fit_bg_image = sigma_clip(image, sigma)
    return fit_model(fit_bg_image, model)


def fit_gaussian2d(image):
    """
    Fit a 2D gaussian to a source in an image.

    Parameters
    ----------
    image : array
        2D array to fit.

    Returns
    -------
    fitted_model : `~astropy.modeling.models.Gaussian2D`
        AstroPy Gaussian2D model with parameters set by the fitter.
    """

    # Estimate center of target
    y_mean, x_mean = np.array(image.shape) // 2  # Center guess

    # Create model to fit
    model = models.Gaussian2D(amplitude=image.max(),
                              x_mean=x_mean,
                              y_mean=y_mean,
                              fixed={}
                              )

    # Fit model to grid
    fitted_model, fit = fit_model(image, model)

    return fitted_model


def print_model_params(model):
    """Print the params and values of an AstroPy model"""
    for param, value in zip(model.param_names, model.parameters):
        print("{:0.4f}\t{}".format(value, param))


def plot_fit(model, image, mode='center', center=None, vmin=None, vmax=None,
             cbar=True, figsize=None, return_images=False, flux_label='Pixel Value'):
    """
    Plot fitted model, its 1D fit profile and residuals.
    If trying to convert a model to image, use `petrofit.modeling.fitting.model_to_image` instead.
    This function dose not call `plt.show()`.

    Parameters
    ----------
    image : array
        2D array that was fit by the model.

    mode : str, optional
        One of the following modes (`astropy.convolution.utils.discretize_model`):
            * ``'center'`` (default)
                Discretize model by taking the value
                at the center of the bin.
            * ``'linear_interp'``
                Discretize model by linearly interpolating
                between the values at the corners of the bin.
                For 2D models interpolation is bilinear.
            * ``'oversample'``
                Discretize model by taking the average
                on an oversampled grid.
            * ``'integrate'``
                Discretize model by integrating the model
                over the bin using `scipy.integrate.quad`.
                Very slow.

    model : `~astropy.modeling.FittableModel`
        Original data that was fitted by the model.

    mode : str, optional
        One of the following modes (`astropy.convolution.utils.discretize_model`):
            * ``'center'`` (default)
                Discretize model by taking the value
                at the center of the bin.
            * ``'linear_interp'``
                Discretize model by linearly interpolating
                between the values at the corners of the bin.
                For 2D models interpolation is bilinear.
            * ``'oversample'``
                Discretize model by taking the average
                on an oversampled grid.
            * ``'integrate'``
                Discretize model by integrating the model
                over the bin using `scipy.integrate.quad`.
                Very slow.

    center : tuple
        (x, y) Coordinate of the center of the image (in pixels).
        The origin of the image is defined as `origin = center - floor_divide(size, 2)`
        (i.e the image will range from (origin -> origin + size)). If None, the origin
        of the image is assumed to be at (0, 0) (i.e `center = floor_divide(size, 2)`).

    vmin : float
        Min plot value

    vmax : float
        Max plot value

    cbar : bool
        Show color-bar if True.

    figsize : tuple
        Figure size, should be (3*size, size).

    return_images : bool
        If true, this function returns a list of subplot Axes,
        Model-image and the residual (input image - model-image).

    flux_label : str
        Label for color-bar.

    Returns
    -------
    axs, model_image, residual_image, cbar : (array of `.axes.Axes`, array, array, cbar)
        If `return_images`, this function returns a list of subplot Axes,
        Model-image and the residual (input image - model-image).
    """

    if isinstance(image, CCDData) or isinstance(image, Cutout2D):
        image = image.data

    # Make Model Image
    # ----------------

    # Set the size of the model image equal to the fitted image
    fitted_image_size = (image.shape[1], image.shape[0])

    # Generate a model image from the model
    model_image = model_to_image(
        model=model,
        size=fitted_image_size,
        mode=mode,
        center=center
    )

    residual_image = image - model_image

    # Plot Model Image
    # ----------------

    fig, axs = plt.subplots(1, 3, figsize=figsize)

    # If vmin and vmax are not provided, compute them
    if vmin is None:
        vmin = min(image.min(), model_image.min(), residual_image.min())
    if vmax is None:
        vmax = max(image.max(), model_image.max(), residual_image.max())

    im0 = axs[0].imshow(image, vmin=vmin, vmax=vmax)
    axs[0].set_title("Data")
    axs[0].set_xlabel("Pixels")
    axs[0].set_ylabel("Pixels")

    axs[1].imshow(model_image, vmin=vmin, vmax=vmax)
    axs[1].set_title("Model")
    axs[1].set_xlabel("Pixels")
    axs[1].set_ylabel("Pixels")

    axs[2].imshow(residual_image, vmin=vmin, vmax=vmax)
    axs[2].set_title("Residual")
    axs[2].set_xlabel("Pixels")
    axs[2].set_ylabel("Pixels")

    fig_cbar = None
    if cbar:
        # Add a single colorbar at the bottom of the subplots
        cbar_ax = fig.add_axes([0.12, 0.08, 0.78, 0.05])
        fig_cbar = fig.colorbar(im0, cax=cbar_ax, aspect=40, orientation='horizontal')
        fig.subplots_adjust(bottom=0.23)
        fig_cbar.ax.set_xlabel(flux_label)

    if return_images:
        return axs, model_image, residual_image, fig_cbar


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