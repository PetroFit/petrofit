import os
from collections import OrderedDict

import numpy as np

from scipy.ndimage import rotate

from astropy.convolution import convolve
from astropy.nddata import block_reduce
from astropy.modeling import FittableModel, Parameter, custom_model


class PSFModel(FittableModel):

    def evaluate(self, *args, **kwargs):
        psf_p = args[-1]
        args = args[:-1]

        x = args[0]
        y = args[1]
        x_size = max([i.max() + 1 for i in [x, y]])

        x_0 = np.round(args[3][0])
        y_0 = np.round(args[4][0])

        factor = self.oversample
        size = x_size * factor
        mid = size // 2

        x_arange = y_arange = np.arange(0.5, size * factor, 1) / factor
        x_grid, y_grid = np.meshgrid(x_arange, y_arange)  # Scale to x_size
        temp_galaxy_image = self._model.evaluate(x_grid, y_grid, *args[self.n_inputs:])

        x_0_over_sampled = np.argmin(np.abs(x_arange - (x_0 * factor)))
        y_0_over_sampled = np.argmin(np.abs(y_arange - (y_0 * factor)))
        temp_galaxy_image[y_0_over_sampled, x_0_over_sampled] = self._model.evaluate(x_0, y_0, *args[self.n_inputs:])

        temp_galaxy_image = block_reduce(temp_galaxy_image, factor) / factor ** 2

        del x_grid, y_grid, x_arange, y_arange
        if self.psf is None:
            return temp_galaxy_image[y.astype(int), x.astype(int)]

        else:
            PSF_2 = rotate(self.psf, psf_p[0], reshape=False)
            return convolve(temp_galaxy_image, PSF_2)[y.astype(int), x.astype(int)]

    @property
    def model(self):

        model = self._model.copy()
        for param in model.param_names:
            setattr(model, param, getattr(self, param))

        fixed = self.fixed
        del fixed['psf_pa']

        bounds = self.bounds
        del bounds['psf_pa']

        model.fixed.update(fixed)
        model.bounds.update(bounds)

        return model

    @staticmethod
    def wrap(model, psf=None, oversample=1):

        # Extract model params
        params = OrderedDict(
            (param_name, Parameter(param_name, default=param_val)) for param_name, param_val in
            zip(model.param_names, model.parameters)
        )

        # Prepare class attributes
        members = OrderedDict([
            ('__module__', '__main__'),
            ('__doc__', 'PSF Wrapped Model\n{}'.format(model.__doc__)),
            ('n_inputs', model.n_inputs),
            ('n_outputs', model.n_outputs),
            ('oversample', oversample),
            ('psf', psf),
            ('_model', model),
        ])

        # Add params to class attributes
        members.update(params)
        members.update({'psf_pa': Parameter('psf_pa', default=0)})

        # Construct new model class
        model_class = type('PSFModel', (PSFModel,), members)

        # Init new model from new model class
        new_model = model_class()

        # Sync model states
        new_model.fixed.update(model.fixed)
        new_model.bounds.update(model.bounds)

        # Return new model
        return new_model

@custom_model
def Nuker2D(x, y, amplitude=1, r_eff=1, x_0=0, y_0=0, a=1, b=2, g=0, ellip=0, theta=0):
    """Two dimensional Nuker2D model"""
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