import os
from collections import OrderedDict

import numpy as np

from scipy.ndimage import rotate

from astropy.convolution import convolve
from astropy.nddata import block_reduce
from astropy.modeling import models, fitting, FittableModel, Parameter, custom_model
from astropy.modeling.optimizers import DEFAULT_ACC, DEFAULT_EPS, DEFAULT_MAXITER

from matplotlib import pyplot as plt


def fit_model(image, model, maxiter=5000, epsilon=DEFAULT_EPS, acc=DEFAULT_ACC):
    """
    Wrapper function to conveniently fit an image to an input model.

    Parameters
    ----------
    image : array
        2D array to fit.

    model : `~astropy.modeling.FittableModel`
        AstroPy model to sample from.

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

    # Fit model to grid
    fitter = fitting.LevMarLSQFitter()
    fitted_model= fitter(model, x_arange, y_arange, z, maxiter=maxiter, epsilon=epsilon, acc=acc)

    return fitted_model, fitter


def model_to_image(x, y, size, model):
    """
    Construct an image from a model.

    Parameters
    ----------
    x : int
        x center of sampling grid.

    y : int
        y center of sampling grid.

    size : int
        Size of sampling pixel grid.

    model : `~astropy.modeling.FittableModel`
        AstroPy model to sample from.

    Returns
    -------
    model_image : array
        2D image of the model.
    """
    y_arange, x_arange = np.mgrid[
                         int(y) - size//2:int(y) + size//2,
                         int(x) - size//2:int(x) + size//2, ]
    return model(x_arange, y_arange)


def fit_plane(image):
    """
    Given an image, fit a 2D plane.

    Parameters
    ----------
    image : array
        2D array to fit.

    Returns
    -------
    model : `~astropy.modeling.models.Planar2D`
        Plane model with best fit params
    """
    model = models.Planar2D(slope_x=0., slope_y=0, intercept=0)

    # Make x and y grid to fit to
    y_arange, x_arange = np.where(~(np.isnan(image)))

    z = image[(y_arange, x_arange)]

    # Fit model to grid
    fit = fitting.LinearLSQFitter()
    fitted_plane = fit(model, x_arange, y_arange, z)

    return fitted_plane, fit


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
