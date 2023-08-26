import numpy as np
import pytest

from astropy.convolution import convolve
from astropy.modeling import models

import petrofit as pf

from matplotlib import pyplot as plt


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


def test_make_grid():
    # Test default arguments
    x, y = pf.make_grid(3)
    expected_x = np.array([[0., 1., 2.],
                           [0., 1., 2.],
                           [0., 1., 2.]])
    expected_y = np.array([[0., 0., 0.],
                           [1., 1., 1.],
                           [2., 2., 2.]])
    assert np.allclose(x, expected_x)
    assert np.allclose(y, expected_y)

    # Test with origin and factor
    x, y = pf.make_grid(3, origin=(4, 5))
    assert np.allclose(x, expected_x + 4)
    assert np.allclose(y, expected_y + 5)

    x, y = pf.make_grid(2, origin=(0.25, 0.25), factor=2)
    expected_x, expected_y = [
        np.array([
            [0., 0.5, 1., 1.5],
            [0., 0.5, 1., 1.5],
            [0., 0.5, 1., 1.5],
            [0., 0.5, 1., 1.5]
        ]),
         np.array([
             [0., 0., 0., 0.],
             [0.5, 0.5, 0.5, 0.5],
             [1., 1., 1., 1.],
             [1.5, 1.5, 1.5, 1.5]
         ])]
    assert np.allclose(x, expected_x)
    assert np.allclose(y, expected_y)

    # Test factor assertion
    with pytest.raises(AssertionError):
        pf.make_grid(3, factor=1.5)

    # Test origin size
    with pytest.raises(IndexError):
        pf.make_grid(3, origin=(1,))


def test_psf_convolved_model_2d_init():
    with pytest.raises(AssertionError):
        pf.PSFConvolvedModel2D(models.Sersic1D())

    with pytest.raises(AssertionError):
        pf.PSFConvolvedModel2D(models.Sersic1D())

    I_e = 1
    n = 1
    base_model = models.Sersic2D(amplitude=I_e, r_eff=5, n=n, x_0=15., y_0=15.)
    psf = pf.model_to_image(models.Gaussian2D(x_mean=15, y_mean=15, x_stddev=5, y_stddev=5), size=30)
    psf /= psf.sum()

    pf.PSFConvolvedModel2D(base_model)
    pf.PSFConvolvedModel2D(base_model, psf=None, oversample=None, psf_oversample=None)
    pf.PSFConvolvedModel2D(base_model, psf=None, oversample=5, psf_oversample=None)

    pf.PSFConvolvedModel2D(base_model, psf=psf, oversample=None, psf_oversample=None)
    pf.PSFConvolvedModel2D(base_model, psf=psf, oversample=5, psf_oversample=None)
    pf.PSFConvolvedModel2D(base_model, psf=psf, oversample=10, psf_oversample=5)
    model = pf.PSFConvolvedModel2D(base_model, psf=psf, oversample=5, psf_oversample=5)

    assert model.oversample == 5
    assert model.psf_oversample == 5

    with pytest.raises(ValueError):
        # psf_oversample provided but PSF is None
        pf.PSFConvolvedModel2D(base_model, psf=None, oversample=5, psf_oversample=5)

    with pytest.raises(ValueError):
        # oversample should be equal to or an integer multiple of psf_oversample
        pf.PSFConvolvedModel2D(base_model, psf=psf, oversample=1, psf_oversample=5)
        pf.PSFConvolvedModel2D(base_model, psf=psf, oversample=9, psf_oversample=5)

    psf_model = pf.PSFConvolvedModel2D(base_model)
    psf_model.psf = psf
    psf_model.oversample = 4
    psf_model.psf_oversample = 4

    psf_model = pf.PSFConvolvedModel2D(base_model)
    psf_model.oversample = 4
    with pytest.raises(ValueError):
        psf_model.psf_oversample = 4

    psf_model = pf.PSFConvolvedModel2D(base_model)
    psf_model.psf = psf
    psf_model.oversample = None
    with pytest.raises(ValueError):
        # oversample should be equal to or an integer multiple of psf_oversample
        psf_model.psf_oversample = 4

    # Additional Tests
    with pytest.raises(TypeError):
        # Test for float oversample value
        pf.PSFConvolvedModel2D(base_model, psf=psf, oversample=5.5, psf_oversample=5)

    with pytest.raises(ValueError):
        # Test for negative oversample value
        pf.PSFConvolvedModel2D(base_model, psf=psf, oversample=-5, psf_oversample=5)

    psf_model = pf.PSFConvolvedModel2D(base_model)
    psf_model.psf = psf
    psf_model.oversample = 4
    psf_model.psf_oversample = 4

    psf_model = pf.PSFConvolvedModel2D(base_model)
    psf_model.psf = psf
    psf_model.oversample = 8
    psf_model.psf_oversample = 4
    assert psf_model._get_psf_factor() == 4
    assert psf_model._get_oversample_factor() == 8
    psf_model.oversample = 16
    assert psf_model._get_oversample_factor() == 16
    psf_model.oversample = (15, 15, 100, 24)
    assert psf_model._get_oversample_factor() == 24
    psf_model.psf_oversample = None
    assert psf_model._get_psf_factor() == 1
    psf_model.oversample = 3
    assert psf_model._get_oversample_factor() == 3
    psf_model.oversample = None
    assert psf_model._get_psf_factor() == psf_model._get_oversample_factor() == 1


def test_psf_sampling(galfit_psf_images, psf_image):
    for psf_index, image in galfit_psf_images.items():
        base_model = models.Sersic2D(
            amplitude=0.020625826413226116,
            r_eff=30,
            n=4,
            x_0=99.,
            y_0=99.,
            ellip=0,
            theta=0.0,
            bounds=pf.get_default_sersic_bounds(),
            fixed={'x_0': False, 'y_0': False, 'n': True, 'r_eff': True, 'ellip': True, 'theta': True}
        )

        PSF = psf_image if psf_index > 0 else None
        psf_oversample = psf_index if psf_index > 0 else None
        model = pf.PSFConvolvedModel2D(base_model, psf=PSF,
                                       oversample=('x_0', 'y_0', 100, psf_oversample * 5 if PSF is not None else 50),
                                       psf_oversample=psf_oversample if PSF is not None else None
                                       )
        model.fixed.update({'psf_pa': True})

        fitted_model, fit = pf.fit_model(image, model, acc=1e-12, )

        *_, residual = pf.plot_fit(fitted_model, image, vmax=0.039)
        assert abs(residual).max() < 0.05
        pf.print_model_params(fitted_model)
        plt.show()