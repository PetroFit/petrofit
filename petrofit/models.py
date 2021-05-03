import os
from collections import OrderedDict

import numpy as np

from scipy.ndimage import rotate
from scipy.special import gammainc, gamma, gammaincinv

from astropy.convolution import convolve
from astropy.nddata import block_reduce
from astropy.modeling import FittableModel, Parameter, custom_model, models

__all__ = [
    'make_grid', 'PSFModel',
    'Nuker2D', 'Moffat2D', 'EllipMoffat2D',
    'sersic_enclosed', 'sersic_enclosed_inv', 'sersic_enclosed_model',
    'petrosian_profile', 'petrosian_model', 
]


def make_grid(size, factor=1):
    assert isinstance(factor, int)
    x_arange = y_arange = (np.arange(0.5, size * factor, 1) / factor) - 0.5
    return np.meshgrid(x_arange, y_arange)


class PSFModel(FittableModel):
    oversample = None

    _cache_grid = True
    _cached_grid_size = 0
    _cached_grid_factor = 0
    _cached_grid = None

    def clear_cached_grid(self):
        self._cached_grid_size = 0
        self._cached_grid_factor = 0
        if self._cached_grid is not None:
            del self._cached_grid
        self._cached_grid = None

    @property
    def cache_grid(self):
        return self._cache_grid

    @cache_grid.setter
    def cache_grid(self, value):
        if value == False:
            self._cache_grid = False
            self.clear_cached_grid()
        elif value == True:
            self._cache_grid = True
        else:
            raise ValueError("{} is not a bool, use True or False".format(value))

    def evaluate(self, *args, **kwargs):
        psf_p = args[-1]
        args = args[:-1]

        x = args[0]
        y = args[1]

        assert not np.any(x < 0), 'negative pixel values not supported at this time'
        assert not np.any(y < 0), 'negative pixel values not supported at this time'

        grid_size = max([i.max() + 1 for i in [x, y]])
        grid_factor = self.oversample if isinstance(self.oversample, int) else 1

        if grid_size == self._cached_grid_size and self._cached_grid_factor == grid_factor:
            main_grid = self._cached_grid
        else:
            main_grid = make_grid(grid_size, factor=grid_factor)

            if self.cache_grid:
                self._cached_grid = main_grid
                self._cached_grid_size = grid_size
                self._cached_grid_factor = grid_factor

        x_grid, y_grid = main_grid

        model_image = self._model.evaluate(x_grid, y_grid, *args[self.n_inputs:])

        if isinstance(self.oversample, int):
            model_image = block_reduce(model_image, grid_factor) / grid_factor ** 2

        elif isinstance(self.oversample, tuple):
            sub_grid_x0, sub_grid_y0, sub_grid_size, sub_grid_factor = self.oversample

            if isinstance(sub_grid_x0, str):
                assert sub_grid_x0 in self._model.param_names, "oversample param '{}' is not in the wrapped model param list".format(
                    sub_grid_x0)

                idx = self._model.param_names.index(sub_grid_x0)
                sub_grid_x0 = args[self.n_inputs:][idx][0]

            if isinstance(sub_grid_y0, str):
                assert sub_grid_y0 in self._model.param_names, "oversample param '{}' is not in the wrapped model param list".format(
                    sub_grid_y0)

                idx = self._model.param_names.index(sub_grid_y0)
                sub_grid_y0 = args[self.n_inputs:][idx][0]

            x_sub_grid, y_sub_grid = make_grid(sub_grid_size, factor=sub_grid_factor)

            x_sub_grid += int(sub_grid_x0) - sub_grid_size // 2
            y_sub_grid += int(sub_grid_y0) - sub_grid_size // 2

            sub_model_oversampled_image = self._model.evaluate(x_sub_grid, y_sub_grid, *args[self.n_inputs:])

            # Experimental  
            # over_sampled_sub_model_x0 = np.argmin(
            #     np.abs(x_sub_grid[0, :] - 1 / (2 * sub_grid_factor) - (sub_grid_x0 * sub_grid_factor)))
            # over_sampled_sub_model_y0 = np.argmin(
            #     np.abs(y_sub_grid[:, 0] - 1 / (2 * sub_grid_factor) - (sub_grid_y0 * sub_grid_factor)))
            #
            # sub_model_oversampled_image[
            #     over_sampled_sub_model_y0,
            #     over_sampled_sub_model_x0
            # ] = self._model.evaluate(sub_grid_x0, sub_grid_y0, *args[self.n_inputs:])

            sub_model_image = block_reduce(sub_model_oversampled_image, sub_grid_factor) / sub_grid_factor ** 2

            x_sub_min = int(x_sub_grid[0][0] - 1 / (2 * sub_grid_factor)) + 1
            y_sub_min = int(y_sub_grid[0][0] - 1 / (2 * sub_grid_factor)) + 1

            model_image[
            y_sub_min: y_sub_min + sub_grid_size,
            x_sub_min: x_sub_min + sub_grid_size
            ] = sub_model_image

        if self.psf is None:
            return model_image[y.astype(int), x.astype(int)]

        else:
            psf = self.psf
            if psf_p[0] != 0:
                psf = rotate(psf, psf_p[0], reshape=False)
            return convolve(model_image, psf)[y.astype(int), x.astype(int)]

    @property
    def model(self):

        model = self._model.copy()
        for param in model.param_names:
            setattr(model, param, getattr(self, param).value)

        fixed = self.fixed
        del fixed['psf_pa']

        bounds = self.bounds
        del bounds['psf_pa']

        model.fixed.update(fixed)
        model.bounds.update(bounds)

        return model

    @staticmethod
    def wrap(model, psf=None, oversample=None):
        if isinstance(model, PSFModel):
            raise TypeError("Can not wrap a PSFModel, try: PSFModel.wrap(psf_model.model)")

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

        # add oversample regions
        if oversample is not None:
            if isinstance(oversample, int) or isinstance(oversample, tuple):
                new_model.oversample = oversample
            else:
                raise ValueError("oversample should be a single int factor or a tuple (x, y, size, factor).")

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


def sersic_enclosed(
        r,
        amplitude,
        r_eff,
        n,
):
    bn = gammaincinv(2. * n, 0.5)
    x = bn * (r / r_eff) ** (1 / n)
    g = gamma(2. * n) * gammainc(2. * n, x)

    return amplitude * (r_eff) ** 2 * 2 * np.pi * n * ((np.exp(bn)) / (bn) ** (2 * n)) * g


def sersic_enclosed_inv(
        f,
        amplitude,
        r_eff,
        n,
):
    bn = gammaincinv(2. * n, 0.5)
    g = f / (amplitude * (r_eff) ** 2 * 2 * np.pi * n * ((np.exp(bn)) / (bn) ** (2 * n)))

    x = gammaincinv(2. * n, g / gamma(2. * n))

    return (x / bn) ** n * r_eff


@custom_model
def sersic_enclosed_model(
        x,
        amplitude=1000,
        r_eff=30,
        n=2,
):
    return sersic_enclosed(x, amplitude, r_eff, n)


def petrosian_profile(r, r_eff, n):
    bn = gammaincinv(2. * n, 0.5)

    x = bn * (r / r_eff) ** (1 / n)

    g = gamma(2 * n) * gammainc(2 * n, x)

    return (np.exp(-x) * x ** (2 * n)) / (2 * n * g)


@custom_model
def petrosian_model(
    x, 
    r_eff=1, 
    n=4

):
    return petrosian_profile(x, r_eff, n)
