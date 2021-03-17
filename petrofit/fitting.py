import os

import numpy as np

from astropy.stats import mad_std, gaussian_sigma_to_fwhm
from astropy.modeling import models, fitting, functional_models, Parameter, custom_model
from astropy.modeling.optimizers import DEFAULT_ACC, DEFAULT_EPS, DEFAULT_MAXITER

from matplotlib import pyplot as plt


def print_model_params(model):
    for param, value in zip(model.param_names, model.parameters):
        print("{:0.4f}\t{}".format(value,param))


def model_to_image(x, y, size, model):
    y_arange, x_arange = np.mgrid[
                         int(y) - size//2:int(y) + size//2,
                         int(x) - size//2:int(x) + size//2, ]
    return model(x_arange, y_arange)


def fit_plane(image):
    model = models.Planar2D(slope_x=0., slope_y=0, intercept=0)

    # Make x and y grid to fit to
    y_arange, x_arange = np.where(~(np.isnan(image)))

    z = image[(y_arange, x_arange)]

    # Fit model to grid
    fit = fitting.LinearLSQFitter()
    fitted_plane = fit(model, x_arange, y_arange, z)

    return fitted_plane, fit


def fit_model(image, model, maxiter=5000, epsilon=DEFAULT_EPS, acc=DEFAULT_ACC):
    # Make x and y grid to fit to
    y_arange, x_arange = np.where(~(np.isnan(image)))

    z = image[(y_arange, x_arange)]

    # Fit model to grid
    fit = fitting.LevMarLSQFitter()
    fitted_line = fit(model, x_arange, y_arange, z, maxiter=maxiter, epsilon=epsilon, acc=acc)

    return fitted_line, fit


def fit_sersic2d(image, ellip=0.5, theta=0, fixed={}):
    # Estimate center of target
    y_mean, x_mean = np.array(image.shape) // 2  # Center guess

    # Create model to fit
    model = models.Sersic2D(amplitude=image.max(),
                            r_eff=x_mean//4,
                            n=2,
                            x_0=x_mean,
                            y_0=y_mean,
                            ellip=ellip,
                            theta=theta,
                            fixed=fixed
                            )

    # Fit model to grid
    fitted_line, fit = fit_model(image, model)

    return fitted_line


def fit_gaussian2d(image):
    # Estimate center of target
    y_mean, x_mean = np.array(image.shape) // 2  # Center guess

    # Create model to fit
    model = models.Gaussian2D(amplitude=image.max(),
                              x_mean=x_mean,
                              y_mean=y_mean,
                              fixed={}
                              )

    # Fit model to grid
    fitted_line, fit = fit_model(image, model)

    return fitted_line


def model_subtract(image, target, x, y):
    dy, dx = target.shape
    bounds = np.array([y - dy // 2, y + dy // 2, x - dx // 2, x + dx // 2])

    ymin, ymax, xmin, xmax = bounds

    targ_xmin = None
    if xmin < 0:
        targ_xmin = abs(xmin)
        xmin = 0

    targ_ymin = None
    if ymin < 0:
        targ_ymin = abs(ymin)
        ymin = 0

    targ_xmax = None
    if xmax >= image.shape[1]:
        targ_xmax = image.shape[1] - x + target.shape[1] // 2
        xmax = image.shape[1]

    targ_ymax = None
    if ymax >= image.shape[0]:
        targ_ymax = image.shape[0] - y + target.shape[0] // 2
        ymax = image.shape[0]

    image[ymin:ymax, xmin:xmax] -= target[targ_ymin:targ_ymax, targ_xmin:targ_xmax]

    return image


def plot_fit(image, model, vmin=None, vmax=None):
    if isinstance(model, models.Sersic2D):
        x_0, y_0 = model.x_0, model.y_0  # Center
    elif isinstance(model, models.Gaussian2D):
        x_0, y_0 = [i.value for i in [model.x_mean, model.y_mean]]
    else:
        x_0, y_0 = model.x_0, model.y_0  # Center

    if isinstance(x_0, Parameter):
        x_0, y_0 = [int(i.value) for i in [x_0, y_0]]

    fig = plt.figure(figsize=(12, 12))

    # Make x and y grid to plot to
    y_arange, x_arange = np.mgrid[:image.shape[0], :image.shape[1]]

    # Plot input image with FWHM and center
    # -------------------------------------
    ax0 = fig.add_subplot(221)

    ax0.imshow(image, vmin=vmin, vmax=vmax)
    ax0.axvline(x_0, label="Center")
    ax0.axhline(y_0)

    ax0.set_title("Image")
    ax0.set_xlabel("X Pixel")
    ax0.set_ylabel("Y Pixel")

    ax0.legend()

    # Plot residuals
    # ---------------

    residuals = image - model(x_arange, y_arange)
    # residuals[np.where(residuals < 0)] = 0.
    ax1 = fig.add_subplot(222)
    ax1.imshow(residuals, vmin=vmin, vmax=vmax)

    ax1.set_title("Residual (Image - Fit)")
    ax1.set_xlabel("X Pixel")
    ax1.set_ylabel("Y Pixel")

    # Prepare fine grid
    # -----------------

    # We need a fine grid to fill in inter-pixel values
    # Oversample by a factor of 10

    y_arange_fine, x_arange_fine = np.mgrid[:image.shape[0] * 10, :image.shape[1] * 10] / 10

    fine_image = model(x_arange_fine, y_arange_fine)
    x_slice_fine = fine_image[fine_image.shape[0] // 2, :]
    y_slice_fine = fine_image[:, fine_image.shape[1] // 2]

    # Plot X fit
    # ----------

    ax2 = fig.add_subplot(223)

    ax2.plot(x_arange_fine[1, :], x_slice_fine, c='r')
    ax2.scatter(x_arange[1, :], image[int(np.round(y_0)), :], c='black')

    ax2.set_title("X Cross Section")
    ax2.set_xlabel("X Pixel")
    ax2.set_ylabel("Flux")

    # Plot Y fit
    # ----------

    ax3 = fig.add_subplot(224)

    ax3.plot(y_arange_fine[:, 1], y_slice_fine, c='r')
    ax3.scatter(y_arange[:, 1], image[:, int(np.round(x_0))], c='black')

    ax3.set_title("Y Cross Section")
    ax3.set_xlabel("Y Pixel")
    ax3.set_ylabel("Flux")

    return fig, [ax0, ax1, ax2, ax3]


@custom_model
def Nuker2D(x, y, amplitude=1, r_eff=1, x_0=0, y_0=0, a=1, b=2, g=0, ellip=0, theta=0):
    A, B = 1 * r_eff, (1 - ellip) * r_eff
    cos_theta, sin_theta = np.cos(theta), np.sin(theta)
    x_maj = (x - x_0) * cos_theta + (y - y_0) * sin_theta
    x_min = -(x - x_0) * sin_theta + (y - y_0) * cos_theta
    r = np.sqrt((x_maj / A) ** 2 + (x_min / B) ** 2)

    return 2 ** ((b - g) / a) * amplitude * (r_eff / r) ** (g) * (1 + (r / r_eff) ** a) ** ((g - b) / a)


@custom_model
def Moffat2D(x, y, amplitude=1.0, x_0=0.0, y_0=0.0, gamma=1.0, alpha=1.0):
    """Two dimensional Moffat function."""
    rr_gg = ((x - x_0) ** 2 + (y - y_0) ** 2) / gamma ** 2
    return amplitude * (1 + rr_gg) ** (-alpha)


@custom_model
def EllipMoffat2D(x, y, amplitude=1.0, x_0=0.0, y_0=0.0, gamma=1.0, alpha=1.0, ellip=0, theta=0, r=1):
    """Two dimensional Moffat function."""

    a, b = 1 * r, (1 - ellip) * r
    cos_theta, sin_theta = np.cos(theta), np.sin(theta)
    x_maj = (x - x_0) * cos_theta + (y - y_0) * sin_theta
    x_min = -(x - x_0) * sin_theta + (y - y_0) * cos_theta
    z = np.sqrt((x_maj / a) ** 2 + (x_min / b) ** 2)

    rr_gg = (z) / gamma ** 2

    return amplitude * (1 + rr_gg) ** (-alpha)
