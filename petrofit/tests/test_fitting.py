import numpy as np

from astropy.convolution import convolve
from astropy.modeling import models

import petrofit as pf


def test_psf_convolved_image_model():
    """Test fitting and PSF convolution"""

    # Make model:
    imsize = 300

    sersic_model = models.Sersic2D(
        amplitude=1,
        r_eff=25,
        n=2,
        x_0=imsize / 2,
        y_0=imsize / 2,
        ellip=0,
        theta=0,
        bounds={
            'amplitude': (0., None),
            'r_eff': (0, None),
            'n': (0, 10),
            'ellip': (0, 1),
            'theta': (-2 * np.pi, 2 * np.pi),
        },
    )

    # Make model image
    image = pf.model_to_image(sersic_model, imsize)

    # Make a PSF
    x_grid, y_grid = pf.make_grid(51, factor=1)
    PSF = models.Moffat2D(x_0=25.0, y_0=25.0)(x_grid, y_grid)
    PSF /= PSF.sum()

    # Make a PSF image using model image and PSF
    psf_sersic_image = convolve(image, PSF)

    # Make a PSFConvolvedModel2D
    psf_sersic_model = pf.PSFConvolvedModel2D(sersic_model, psf=PSF, oversample=None)
    psf_sersic_model.fixed['psf_pa'] = True

    # Make a PSFConvolvedModel2D image
    psf_sersic_model_image = pf.model_to_image(psf_sersic_model, imsize)

    # Compare the PSF image to PSFConvolvedModel2D image
    error_arr = abs(psf_sersic_model_image - psf_sersic_image) / psf_sersic_image
    assert np.max(error_arr) < 0.01  # max error less than 1% error

    # Test fitting

    # Change params with offset
    psf_sersic_model.r_eff = 23
    psf_sersic_model.x_0 = 0.6 + imsize / 2
    psf_sersic_model.y_0 = 0.1 + imsize / 2
    psf_sersic_model.n = 3

    # Fit
    fitted_model, fit_info = pf.fit_model(
        psf_sersic_image, psf_sersic_model,
        maxiter=10000,
        epsilon=1.4901161193847656e-10,
        acc=1e-9,
    )

    # Generate a model image from the fitted model
    fitted_model_image = pf.model_to_image(fitted_model, imsize)

    # Check if fit is close to actual
    error_arr = abs(fitted_model_image - psf_sersic_image) / psf_sersic_image
    assert np.max(error_arr) < 0.01  # max error less than 1% error
