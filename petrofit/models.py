import os
from collections import OrderedDict
import warnings

import numpy as np

from scipy.ndimage import rotate
from scipy.special import gammainc, gamma, gammaincinv
from scipy.signal import convolve

from astropy.nddata import block_reduce
from astropy.modeling import FittableModel, Parameter, custom_model, models

__all__ = [
    'get_default_sersic_bounds', 'make_grid', 'PSFConvolvedImageModel',
    'Nuker2D', 'Moffat2D', 'EllipMoffat2D',
    'sersic_enclosed', 'sersic_enclosed_inv', 'sersic_enclosed_model',
    'petrosian_profile', 'petrosian_model',
]


def get_default_sersic_bounds():
    """Returns the default bounds of the Sersic profile."""
    bounds = {
        'amplitude': (0., None),
        'r_eff': (0, None),
        'n': (0, 10),
        'ellip': (0, 1),
        'theta': (-2 * np.pi, 2 * np.pi),
    }
    return bounds


def make_grid(size, origin=(0, 0), factor=1):
    assert isinstance(factor, int)
    x_arange = (np.arange(0.5, size * factor, 1) / factor) - 0.5
    y_arange = (np.arange(0.5, size * factor, 1) / factor) - 0.5
    x_arange += origin[0]
    y_arange += origin[1]
    return np.meshgrid(x_arange, y_arange)


class PSFModel(FittableModel):
    """
    PSFModel is deprecated as of v0.4.0 and will be
    removed in the next release, use `PSFConvolvedImageModel` instead.
    """
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
            return convolve(model_image, psf, mode='same')[y.astype(int), x.astype(int)]

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
        warnings.warn('''PSFModel is deprecated as of v0.4.0 and will be 
        removed in the next release, use `PSFConvolvedImageModel` instead.''', DeprecationWarning, stacklevel=2)

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
            ('__name__', 'PSFModel'),
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


class PSFConvolvedImageModel(FittableModel):
    """
    Fittable model for converting `FittableModel`s and `CompoundModel`s into 2D images.
    This model takes the input sub-model and adds PSF convolution, as well as PSF convolution.

    Parameters
    ----------
    model : `astropy.modeling.core.Model`
        Base model to convert into an image.

    psf : array
        2D normalized (i.e sum(psf) = 1) image of the point spread function.

    oversample : None or int or tuple
        Oversampling factor. If set to None, no oversampling will be applied to the image.
         If an integer is provided, the whole image will be oversampled by that factor.
         If a tuple of `(center_x, center_y, box_length, oversample_factor)` can be used to
         define an oversampling window. `box_length` and  `oversample_factor` should always be
         integers. `center_x` and  `center_y` can be either float values of the oversampling window or
         string names of parameters in the input model (for example `"x_0"`).

    name : string
        Name for the `PSFConvolvedImageModel` model instance.
    """

    # Default _param_names list; this will be filled in by the implementation's
    # __init__
    _param_names = ()

    n_inputs = 2  # Model has 2 inputs (x, y)
    n_outputs = 1  # Model has 1 output z where z = Z(x, y)

    oversample = None  # Oversampling factor
    _psf = None  # PSF image

    _cache_grid = True  # Cache sampling grid?
    _cached_grid_size = 0  # Cached sampling grid size
    _cached_grid_factor = 0  # Cached sampling grid oversampling rule
    _cached_grid = None  # Cached sampling grid
    _cache_grid_range = None  # Cached sampling grid range (x.min, x.max, y.min, y.max)

    def __init__(self, model, psf=None, oversample=None, name=None, **kwargs):
        # Reset params
        self._parameters = None
        self._parameters_ = {}

        # Check if input model is good
        if isinstance(model, self.__class__):
            raise TypeError("Can not wrap a PSFConvolvedModel2D, try: PSFConvolvedModel2D(psf_convolved_model.model)")
        assert model.n_inputs == 2, "Input model is not 2D"

        # Save attributes
        self.psf = psf
        self._model = model

        # Load model params
        self._load_parameters()

        # add oversample regions
        if oversample is not None:
            if isinstance(oversample, int) or isinstance(oversample, tuple):
                self.oversample = oversample
            else:
                raise ValueError("oversample should be a single int factor or a tuple (x, y, size, factor).")

        super().__init__(name=name, **kwargs)

        # Sync model states
        if 'fixed' not in kwargs:
            self.fixed.update(model.fixed)
        if 'bounds' not in kwargs:
            self.bounds.update(model.bounds

    def _load_parameters(self):
        """Function to load parameters from the sub-model"""
        if self._parameters is not None:
            # do nothing
            return

        self._parameters_ = {}
        self._param_names = []

        for param_name, param_val in zip(self._model.param_names, self._model.parameters):
            param = Parameter(param_name, default=param_val)
            self.__dict__[param_name] = param
            self._parameters_[param_name] = param
            self._param_names.append(param_name)

        param_name = 'psf_pa'
        param = Parameter(param_name, default=0)
        self.__dict__[param_name] = param
        self._parameters_[param_name] = param
        self._param_names.append(param_name)

        self._param_names = tuple(self._param_names)

    @property
    def model(self):
        """Returns sub-model with current parameters of the `PSFConvolvedImageModel`"""

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

    @property
    def psf(self):
        """PSF Image"""
        return self._psf

    @psf.setter
    def psf(self, psf):
        if psf is not None and psf.sum() != 1:
            warnings.warn("Input PSF not normalized to 1")
        self._psf = psf

    def clear_cached_grid(self):
        """Clears cached grid and resets class attributes to default values"""
        self._cached_grid_size = 0
        self._cached_grid_factor = 0
        if self._cached_grid is not None:
            del self._cached_grid
        self._cached_grid = None
        self._cache_grid_range = None

    @property
    def cache_grid(self):
        """Returns the cached sampling grid"""
        return self._cache_grid

    @cache_grid.setter
    def cache_grid(self, value):
        """Sets the cached sampling grid"""
        if value == False:
            self._cache_grid = False
            self.clear_cached_grid()
        elif value == True:
            self._cache_grid = True
        else:
            raise ValueError("{} is not a bool, use True or False".format(value))

    def evaluate(self, x, y, *params, **kwargs):

        # Extract sub-model params as well as `psf_p`
        *sub_model_params, psf_p = params

        # Prepare image indices
        i = (x - x.min()).astype(int)
        j = (y - y.min()).astype(int)

        # Compute image size and oversampling factor
        grid_size = max([i.max(), j.max()]) + 1
        grid_factor = self.oversample if isinstance(self.oversample, int) else 1
        grid_range = (x.min(), x.max(), y.min(), y.max())

        # Make the main grid
        if grid_size == self._cached_grid_size and self._cached_grid_factor == grid_factor and grid_range == self._cache_grid_range:
            # Check if the gird cached
            main_grid = self._cached_grid
        else:
            # Else make a gird
            main_grid = make_grid(grid_size, origin=(x.min(), y.min()), factor=grid_factor)

            if self.cache_grid:
                self._cached_grid = main_grid
                self._cached_grid_size = grid_size
                self._cached_grid_factor = grid_factor
                self._cache_grid_range = grid_range

        # Split main grid to x and y components
        x_grid, y_grid = main_grid

        # Main Model Image
        # ----------------
        # Construct main model image by sampling sub-model
        model_image = self._model.evaluate(x_grid, y_grid, *sub_model_params)

        # Oversampling
        # ------------
        if isinstance(self.oversample, int):
            # If the oversample factor is an int, block reduce the image
            model_image = block_reduce(model_image, grid_factor) / grid_factor ** 2

        elif isinstance(self.oversample, tuple):
            # If the oversample is a window, compute pixel values for that window

            # Load the window params
            sub_grid_x0, sub_grid_y0, sub_grid_size, sub_grid_factor = self.oversample

            assert isinstance(sub_grid_size, int), "Oversampling window size must be an int"
            assert isinstance(sub_grid_factor, int), "Oversampling factor must be an int"

            # If the center of the window is a parameter name, extract its value
            if isinstance(sub_grid_x0, str):
                assert sub_grid_x0 in self._model.param_names, "oversample param '{}' is not in the wrapped model param list".format(
                    sub_grid_x0)

                idx = self._model.param_names.index(sub_grid_x0)
                sub_grid_x0 = sub_model_params[idx][0]

            if isinstance(sub_grid_y0, str):
                assert sub_grid_y0 in self._model.param_names, "oversample param '{}' is not in the wrapped model param list".format(
                    sub_grid_y0)

                idx = self._model.param_names.index(sub_grid_y0)
                sub_grid_y0 = sub_model_params[idx][0]

            # Compute the corner of the sub-grid
            sub_grid_origin = (sub_grid_x0 - sub_grid_size // 2, sub_grid_y0 - sub_grid_size // 2)

            # Make an oversampled sub-grid for window
            x_sub_grid, y_sub_grid = make_grid(sub_grid_size, origin=sub_grid_origin, factor=sub_grid_factor)

            # Sample the sub-model onto the sub-grid
            sub_model_oversampled_image = self._model.evaluate(x_sub_grid, y_sub_grid, *sub_model_params)

            # Block reduce the window to the main image resolution
            sub_model_image = block_reduce(sub_model_oversampled_image, sub_grid_factor) / sub_grid_factor ** 2

            # Compute window indices in main image frame
            i_sub_min = int(np.round(sub_grid_origin[0]))
            j_sub_min = int(np.round(sub_grid_origin[1]))
            i_sub_max = i_sub_min + sub_grid_size
            j_sub_max = j_sub_min + sub_grid_size

            # Clip window indices
            if i_sub_min < 0:
                i_sub_min = 0
            if j_sub_min < 0:
                j_sub_min = 0
            if i_sub_max > i.max():
                i_sub_max = i.max() + 1
            if j_sub_max > j.max():
                j_sub_max = j.max() + 1

            # Add oversampled window to image
            model_image[
            j_sub_min:j_sub_max,
            i_sub_min:i_sub_max
            ] = sub_model_image

        # PSF convolve
        # ------------
        if self.psf is None:
            return model_image[j, i]
        else:
            psf = self.psf
            if psf_p[0] != 0:
                psf = rotate(psf, psf_p[0], reshape=False)
            return convolve(model_image, psf, mode='same')[j, i]

    @property
    def param_names(self):
        """
        On most `Model` classes this is a class attribute, but for `PSFConvolvedImageModel`
        models it is an instance attribute since each input sub-model
        can have different parameters.
        """
        return self._param_names


@custom_model
def Nuker2D(x, y, amplitude=1, r_break=1, x_0=0, y_0=0,
            alpha=2, betta=4, gamma=0):

    """Two dimensional Nuker2D model"""
    x_maj = (x - x_0)  + (y - y_0)
    x_min = -(x - x_0) + (y - y_0)
    r = np.sqrt((x_maj) ** 2 + (x_min) ** 2)

    r[np.where(r == 0)] = np.nan

    return amplitude *  (2 ** ((betta - gamma) / alpha)) * (r / r_break) ** (- gamma) * (1 + (r / r_break) ** alpha) ** ((gamma - betta) / alpha)


@custom_model
def CoreSersic2D(x, y, amplitude=1, r_eff=1, r_break=1, n=1, x_0=0, y_0=0,
                 alpha=10, gamma=0.1, ellip=0, theta=0):
    """
    Core Sersic model as defined by Graham et al 2003.
    """
    bn = gammaincinv(2. * n, 0.5)
    a, b = r_eff, (1 - ellip) * r_eff
    cos_theta, sin_theta = np.cos(theta), np.sin(theta)
    x_maj = (x - x_0) * cos_theta + (y - y_0) * sin_theta
    x_min = -(x - x_0) * sin_theta + (y - y_0) * cos_theta
    r = np.sqrt((x_maj / a) ** 2 + (x_min / b) ** 2)

    r[np.where(r == 0)] = np.nan

    I = amplitude * (2 ** -(gamma / alpha)) * np.exp(
        bn * (2 ** (1 / alpha) * (r_break / r_eff)) ** (1 / n)
    )

    return I * (1 + (r_break / r) ** (alpha)) ** (gamma / alpha) * np.exp(
        - bn * ((r ** alpha + r_break ** alpha) / (r_eff) ** alpha) ** (1 / (alpha * n))
    )


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
        ellip=0
):
    bn = gammaincinv(2. * n, 0.5)
    x = bn * (r / r_eff) ** (1 / n)
    g = gamma(2. * n) * gammainc(2. * n, x)

    return amplitude * (r_eff ** 2) * 2 * np.pi * n * ((np.exp(bn)) / (bn) ** (2 * n)) * g * (1 - ellip)


def sersic_enclosed_inv(
        f,
        amplitude,
        r_eff,
        n,
        ellip=0
):
    bn = gammaincinv(2. * n, 0.5)
    g = f / (amplitude * (r_eff) ** 2 * 2 * np.pi * n * ((np.exp(bn)) / (bn) ** (2 * n)) * (1 - ellip))

    x = gammaincinv(2. * n, g / gamma(2. * n))

    return (x / bn) ** n * r_eff


@custom_model
def sersic_enclosed_model(
        x,
        amplitude=1000,
        r_eff=30,
        n=2,
        ellip=0,
):
    return sersic_enclosed(x, amplitude, r_eff, n, ellip=ellip)


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
