import os
from copy import copy
from collections import OrderedDict
import warnings

import numpy as np

from scipy.ndimage import rotate
from scipy.special import gammainc, gamma, gammaincinv
from scipy.signal import convolve

from astropy.nddata import block_reduce
from astropy.modeling import FittableModel, Parameter, custom_model, models

__all__ = [
    "get_default_sersic_bounds",
    "make_grid",
    "PSFConvolvedModel2D",
    "Nuker2D",
    "sersic_enclosed",
    "sersic_enclosed_inv",
    "sersic_enclosed_model",
    "PetroApprox",
    "petrosian_profile",
    "petrosian_model",
    "get_default_gen_sersic_bounds",
    "GenSersic2D",
    "Ie_to_I0",
    "I0_to_Ie",
]


def Ie_to_I0(I_e, n):
    """
    Converts the sersic intensity at the effective radius (I_e) to the intensity at the 0 radius (I_0)
    based on the sersic index (n)

    Parameters
    ----------
    I_e : float
        Sersic intensity at the effective radius.
    n : float
        Sersic index.

    Returns
    -------
    float
        Intensity at the 0 radius (I_0)
    """
    return I_e * np.exp(gammaincinv(2.0 * n, 0.5))


def I0_to_Ie(I_0, n):
    """
    Converts the intensity at the 0 radius (I_0) to the Sersic intensity at the effective radius (I_e)
    based on the Sersic index (n).

    Parameters
    ----------
    I_0 : float
        Intensity at the 0 radius (I_0).
    n : float
        Sersic index.

    Returns
    -------
    float
        Sersic intensity at the effective radius (I_e).
    """
    return I_0 / np.exp(gammaincinv(2.0 * n, 0.5))


def get_default_sersic_bounds(override={}):
    """Returns the default bounds of the Sersic profile."""
    bounds = {
        "amplitude": (0.0, None),
        "r_eff": (1e-3, None),
        "n": (0.1, 10),
        "ellip": (0, 0.99),
        "theta": (-2 * np.pi, 2 * np.pi),
    }
    bounds.update(override)
    return bounds


def get_default_gen_sersic_bounds(override={}):
    """
    Returns the default bounds of a Generalized Sersic profile.
    This is identical to `get_default_sersic_bounds` but adds a `c_0` bound.
    """
    bounds = get_default_sersic_bounds()
    bounds["c_0"] = (0, 2.0)
    bounds.update(override)
    return bounds


def make_grid(size, origin=(0, 0), factor=1):
    """
    Function to make image sampling grid.

    Parameters
    ----------
    size : int
        The size of the sampling grid. Sampling grid is a square with each side of size `size`.

    origin : tuple
        Bottom corner of the sampling grid (x, y).

    factor : int
        Oversampling factor
    """
    assert isinstance(factor, int)
    x_arange = (np.arange(0.5, size * factor, 1) / factor) - 0.5
    y_arange = (np.arange(0.5, size * factor, 1) / factor) - 0.5
    x_arange += origin[0]
    y_arange += origin[1]
    return np.meshgrid(x_arange, y_arange)


class PSFConvolvedModel2D(FittableModel):
    """
    Fittable model for converting `FittableModel` and `CompoundModel` into 2D images.
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

    psf_oversample : None or int
        Oversampling factor of the PSF relative to data. The `oversample` factor should be an integer multiple
        of the PSF oversampling factor (i.e `oversample > psf_oversample`).

    name : string
        Name for the `PSFConvolvedModel2D` model instance.
    """

    # Default _param_names list; this will be filled in by the implementation's
    # __init__
    _param_names = ()

    n_inputs = 2  # Model has 2 inputs (x, y)
    n_outputs = 1  # Model has 1 output z where z = Z(x, y)

    _oversample = None  # Oversampling factor
    _psf_oversample = None  # PSF oversampling factor relative to image
    _psf = None  # PSF image

    _cache_grid = True  # Cache sampling grid?
    _cached_grid_size = 0  # Cached sampling grid size
    _cached_grid_factor = 0  # Cached sampling grid oversampling rule
    _cached_grid = None  # Cached sampling grid
    _cache_grid_range = None  # Cached sampling grid range (x.min, x.max, y.min, y.max)

    def __init__(
        self, model, psf=None, oversample=None, psf_oversample=None, name=None, **kwargs
    ):
        # Reset params
        self._parameters = None
        self._parameters_ = {}

        # Check if input model is good
        if isinstance(model, self.__class__):
            raise TypeError(
                "Can not wrap a PSFConvolvedModel2D, try: PSFConvolvedModel2D(psf_convolved_model.model)"
            )
        assert model.n_inputs == 2, "Input model is not 2D"

        # Save attributes
        self.psf = psf
        self._model = model

        # Load model params
        self._load_parameters()

        # add oversample regions
        self.oversample = oversample
        self.psf_oversample = psf_oversample

        super().__init__(name=name, **kwargs)

        # Sync model states
        if "fixed" not in kwargs:
            self.fixed.update(model.fixed)
        if "bounds" not in kwargs:
            self.bounds.update(model.bounds)

    def _load_parameters(self):
        """Function to load parameters from the sub-model"""
        if self._parameters is not None:
            # do nothing
            return

        self._parameters_ = {}
        self._param_names = []

        for param_name, param_val in zip(
            self._model.param_names, self._model.parameters
        ):
            param = Parameter(param_name, default=param_val)
            self.__dict__[param_name] = param
            self._parameters_[param_name] = param
            self._param_names.append(param_name)

        param_name = "psf_pa"
        param = Parameter(param_name, default=0)
        self.__dict__[param_name] = param
        self._parameters_[param_name] = param
        self._param_names.append(param_name)

        self._param_names = tuple(self._param_names)

    @property
    def model(self):
        """Returns sub-model with current parameters of the `PSFConvolvedModel2D`"""

        model = self._model.copy()
        for param in model.param_names:
            setattr(model, param, getattr(self, param).value)

        fixed = copy(self.fixed)
        del fixed["psf_pa"]

        bounds = copy(self.bounds)
        del bounds["psf_pa"]

        model.fixed.update(fixed)
        model.bounds.update(bounds)

        return model

    @property
    def param_names(self):
        """
        On most `Model` classes this is a class attribute, but for `PSFConvolvedModel2D`
        models it is an instance attribute since each input sub-model
        can have different parameters.
        """
        return self._param_names

    @property
    def psf(self):
        """PSF Image"""
        return self._psf

    @psf.setter
    def psf(self, psf):
        if psf is not None and np.round(psf.sum(), 6) != 1:
            warnings.warn(
                "Input PSF not normalized to 1, current sum = {}".format(psf.sum())
            )
        self._psf = psf

    @property
    def oversample(self):
        """Sampling grid oversample Factor"""
        return self._oversample

    @oversample.setter
    def oversample(self, oversample):
        psf_oversample = self._get_psf_factor()
        if oversample is not None:
            if isinstance(oversample, (tuple, list, np.ndarray)):
                if len(oversample) != 4:
                    raise ValueError("oversample should be (x, y, size, factor).")
                if not isinstance(oversample[2], int) or not isinstance(
                    oversample[3], int
                ):
                    raise ValueError("size and factor should be integers.")
                grid_factor = oversample[3]
                oversample = tuple(oversample)  # Important Conversion!
            elif isinstance(oversample, int):
                grid_factor = oversample
            else:
                raise TypeError(
                    "oversample should be a single int factor or a tuple (x, y, size, factor)."
                )
            if grid_factor <= 0:
                raise ValueError("oversample should be a positive int factor.")
            if grid_factor % psf_oversample != 0:
                raise ValueError(
                    "oversample should be equal to or an integer multiple of psf_oversample"
                )
        self._oversample = oversample

    @property
    def psf_oversample(self):
        """PSF oversample factor relative to data"""
        return self._psf_oversample

    @psf_oversample.setter
    def psf_oversample(self, psf_oversample):
        grid_factor = self._get_oversample_factor()
        if psf_oversample is not None:
            if self.psf is None:
                raise ValueError("psf_oversample provided but PSF is None")
            if not isinstance(psf_oversample, int) or psf_oversample <= 0:
                raise TypeError(
                    "psf_oversample should be a single positive int factor."
                )
            if grid_factor % psf_oversample != 0:
                raise ValueError(
                    "oversample should be equal to or an integer multiple of psf_oversample. "
                    "Set PSFConvolvedModel2D.oversample value first."
                )
        self._psf_oversample = psf_oversample

    def _get_oversample_factor(self):
        if isinstance(self.oversample, (tuple, list, np.ndarray)):
            return self.oversample[3]
        else:
            return 1 if self.oversample is None else self.oversample

    def _get_psf_factor(self):
        return 1 if self.psf_oversample is None else self.psf_oversample

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
        if value is False:
            self._cache_grid = False
            self.clear_cached_grid()
        elif value is True:
            self._cache_grid = True
        else:
            raise ValueError("{} is not a bool, use True or False".format(value))

    def evaluate(self, x, y, *params, **kwargs):

        # Extract sub-model params as well as `psf_p`
        *sub_model_params, psf_p = params

        # Prepare image indices
        i = (x - x.min()).astype(int)
        j = (y - y.min()).astype(int)

        # Is oversampling sub-grid based
        is_subgrid_oversample = isinstance(self.oversample, tuple)

        # Compute image size and oversampling factor
        # ------------------------------------------
        psf_factor = (
            self._get_psf_factor()
        )  # Oversampling factor of PSF compared to data

        # Grid oversample factor compared to data:
        # If the oversampling is subgrid, then the main grid is at PSF oversample.
        # If the oversampling is an int, the main grid is at oversample factor.
        grid_factor = (
            psf_factor if is_subgrid_oversample else self._get_oversample_factor()
        )

        grid_to_psf_factor = (
            grid_factor // psf_factor
        )  # Oversampling factor of grid compared to PSF

        # Grid size params:
        grid_size = max([i.max(), j.max()]) + 1  # size = max_index + 1
        grid_range = (x.min(), x.max(), y.min(), y.max())  # Main grid bounds

        # Make main grid
        # --------------
        # Make the main sampling grid
        if (
            grid_size == self._cached_grid_size
            and grid_factor == self._cached_grid_factor
            and grid_range == self._cache_grid_range
        ):
            # If the sampling gird cached
            main_grid = self._cached_grid
        else:
            # Else make a sampling gird
            main_grid = make_grid(
                grid_size, origin=(x.min(), y.min()), factor=grid_factor
            )
            # Cache Grid
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
        if not is_subgrid_oversample and grid_to_psf_factor > 1:
            # If the oversample factor is an int, block reduce the image to PSF resolution
            model_image = (
                block_reduce(model_image, grid_to_psf_factor) / grid_to_psf_factor**2
            )

        elif is_subgrid_oversample:
            # If the oversample is a window, compute pixel values for that window
            # and block reduce the image to PSF resolution.

            # Load the window params
            sub_grid_x0, sub_grid_y0, sub_grid_size, sub_grid_factor = self.oversample
            sub_grid_to_psf_factor = sub_grid_factor // psf_factor

            if sub_grid_to_psf_factor > 1:
                # If the center of the window is a parameter name, extract its value
                if isinstance(sub_grid_x0, str):
                    assert (
                        sub_grid_x0 in self._model.param_names
                    ), "oversample param '{}' is not in the wrapped model param list".format(
                        sub_grid_x0
                    )
                    idx = self._model.param_names.index(sub_grid_x0)
                    sub_grid_x0 = sub_model_params[idx][0] if isinstance(sub_model_params[idx], (list, np.ndarray)) else sub_model_params[idx]

                if isinstance(sub_grid_y0, str):
                    assert (
                        sub_grid_y0 in self._model.param_names
                    ), "oversample param '{}' is not in the wrapped model param list".format(
                        sub_grid_y0
                    )
                    idx = self._model.param_names.index(sub_grid_y0)
                    sub_grid_y0 = sub_model_params[idx][0] if isinstance(sub_model_params[idx], (list, np.ndarray)) else sub_model_params[idx]

                # Compute the corner of the sub-grid
                sub_grid_origin = (
                    np.round(sub_grid_x0) - sub_grid_size // 2,
                    np.round(sub_grid_y0) - sub_grid_size // 2,
                )

                # Make an oversampled sub-grid for window
                x_sub_grid, y_sub_grid = make_grid(
                    sub_grid_size, origin=sub_grid_origin, factor=sub_grid_factor
                )

                # Sample the sub-model onto the sub-grid
                sub_model_oversampled_image = self._model.evaluate(
                    x_sub_grid, y_sub_grid, *sub_model_params
                )

                # Block reduce the window to the psf resolution
                sub_model_image = (
                    block_reduce(sub_model_oversampled_image, sub_grid_to_psf_factor)
                    / sub_grid_to_psf_factor**2
                )

                # Compute window indices in main image frame at data resolution first
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

                # Convert to PSF resolution
                i_sub_min *= psf_factor
                j_sub_min *= psf_factor
                i_sub_max *= psf_factor
                j_sub_max *= psf_factor

                # Add oversampled window to image
                model_image[j_sub_min:j_sub_max, i_sub_min:i_sub_max] = sub_model_image

        # PSF convolve
        # ------------
        if self.psf is not None:
            psf = self.psf
            
            psf_p = psf_p[0] if isinstance(psf_p, (list, np.ndarray)) else psf_p 
            if psf_p != 0:
                psf = rotate(psf, psf_p, reshape=False)
            model_image = convolve(model_image, psf, mode="same")

        if psf_factor > 1:
            # If PSF is oversampled relative to the data, block_reduce to data resolution
            model_image = block_reduce(model_image, psf_factor) / psf_factor**2

        return model_image[j, i]


class GenSersic2D(models.Sersic2D):
    """
    Two dimensional Sersic surface brightness profile with
    Generalized Ellipses described in Peng et al. 2010.


    Parameters
    ----------
    amplitude : float
        Surface brightness at r_eff.
    r_eff : float
        Effective (half-light) radius
    n : float
        Sersic Index.
    x_0 : float, optional
        x position of the center.
    y_0 : float, optional
        y position of the center.
    ellip : float, optional
        Ellipticity.
    theta : float or `~astropy.units.Quantity`, optional
        The rotation angle as an angular quantity
        (`~astropy.units.Quantity` or `~astropy.coordinates.Angle`)
        or a value in radians (as a float). The rotation angle
        increases counterclockwise from the positive x axis.
    c_0 : float
        Boxiness of elliptical isophote.
    """

    c_0 = Parameter(default=0, description="General boxiness of isophote")

    @classmethod
    def evaluate(cls, x, y, amplitude, r_eff, n, x_0, y_0, ellip, theta, c_0):
        """Two dimensional Sersic profile function."""

        if cls._gammaincinv is None:
            from scipy.special import gammaincinv

            cls._gammaincinv = gammaincinv

        bn = cls._gammaincinv(2.0 * n, 0.5)
        a, b = r_eff, (1 - ellip) * r_eff
        cos_theta, sin_theta = np.cos(theta), np.sin(theta)
        x_maj = (x - x_0) * cos_theta + (y - y_0) * sin_theta
        x_min = -(x - x_0) * sin_theta + (y - y_0) * cos_theta

        z = (abs(x_maj / a) ** (c_0 + 2) + abs(x_min / b) ** (c_0 + 2)) ** (
            1 / (c_0 + 2)
        )

        return amplitude * np.exp(-bn * (z ** (1 / n) - 1))


@custom_model
def Nuker2D(x, y, amplitude=1, r_break=1, x_0=0, y_0=0, alpha=2, betta=4, gamma=0):
    """Two dimensional Nuker2D model"""
    x_maj = (x - x_0) + (y - y_0)
    x_min = -(x - x_0) + (y - y_0)
    r = np.sqrt((x_maj) ** 2 + (x_min) ** 2)

    r[np.where(r == 0)] = np.nan

    return (
        amplitude
        * (2 ** ((betta - gamma) / alpha))
        * (r / r_break) ** (-gamma)
        * (1 + (r / r_break) ** alpha) ** ((gamma - betta) / alpha)
    )


@custom_model
def CoreSersic2D(
    x,
    y,
    amplitude=1,
    r_eff=1,
    r_break=1,
    n=1,
    x_0=0,
    y_0=0,
    alpha=10,
    gamma=0.1,
    ellip=0,
    theta=0,
):
    """
    Core Sersic model as defined by Graham et al 2003.
    """
    bn = gammaincinv(2.0 * n, 0.5)
    a, b = r_eff, (1 - ellip) * r_eff
    cos_theta, sin_theta = np.cos(theta), np.sin(theta)
    x_maj = (x - x_0) * cos_theta + (y - y_0) * sin_theta
    x_min = -(x - x_0) * sin_theta + (y - y_0) * cos_theta
    r = np.sqrt((x_maj / a) ** 2 + (x_min / b) ** 2)

    r[np.where(r == 0)] = np.nan

    I = (
        amplitude
        * (2 ** -(gamma / alpha))
        * np.exp(bn * (2 ** (1 / alpha) * (r_break / r_eff)) ** (1 / n))
    )

    return (
        I
        * (1 + (r_break / r) ** (alpha)) ** (gamma / alpha)
        * np.exp(
            -bn * ((r**alpha + r_break**alpha) / (r_eff) ** alpha) ** (1 / (alpha * n))
        )
    )


def sersic_enclosed(r, amplitude, r_eff, n, ellip=0):
    """Total Sersic flux enclosed within a radius."""
    bn = gammaincinv(2.0 * n, 0.5)
    x = bn * (r / r_eff) ** (1 / n)
    g = gamma(2.0 * n) * gammainc(2.0 * n, x)

    return (
        amplitude
        * (r_eff**2)
        * 2
        * np.pi
        * n
        * ((np.exp(bn)) / (bn) ** (2 * n))
        * g
        * (1 - ellip)
    )


def sersic_enclosed_inv(f, amplitude, r_eff, n, ellip=0):
    """Radius that would enclose the input flux."""
    bn = gammaincinv(2.0 * n, 0.5)
    g = f / (
        amplitude
        * (r_eff) ** 2
        * 2
        * np.pi
        * n
        * ((np.exp(bn)) / (bn) ** (2 * n))
        * (1 - ellip)
    )

    x = gammaincinv(2.0 * n, g / gamma(2.0 * n))

    return (x / bn) ** n * r_eff


@custom_model
def sersic_enclosed_model(x, amplitude=1000, r_eff=30, n=2, ellip=0):
    """Model for total Sersic flux enclosed within a radius."""
    return sersic_enclosed(x, amplitude, r_eff, n, ellip=ellip)


def petrosian_profile(r, r_eff, n):
    """Ideal Sersic Petrosian profile evaluated at input radii."""
    bn = gammaincinv(2.0 * n, 0.5)

    x = bn * (r / r_eff) ** (1 / n)

    g = gamma(2 * n) * gammainc(2 * n, x)

    return (np.exp(-x) * x ** (2 * n)) / (2 * n * g)


@custom_model
def petrosian_model(x, r_eff=1, n=4):
    """Ideal Sersic Petrosian model."""
    return petrosian_profile(x, r_eff, n)


class PetroApprox:
    """
    This class contains approximations of various Pertorisian and
    Sersic parameters. These approximations do not take into account
    PSF smearing but can be used to approximate initial guesses.
    """

    @staticmethod
    def u2080_to_c2080(u2080):
        """Uncorrected C2080 to epsilon corrected C2080"""
        return models.Polynomial1D(
            6,
            c0=2.26194802,
            c1=-3.61130833,
            c2=3.8219758,
            c3=-1.6414660,
            c4=0.38059409,
            c5=-0.0450384,
            c6=0.00221922,
        )(u2080)

    @staticmethod
    def c2080_to_n(c2080):
        """Corrected C2080 to Sersic index n"""

        return models.Polynomial1D(
            5,
            c0=-0.41844073,
            c1=0.20487513,
            c2=0.08626531,
            c3=0.0106707,
            c4=-0.00082523,
            c5=0.00002486,
        )(c2080)

    @staticmethod
    def n_to_epsilon(n):
        """Sersic index n to epsilon"""
        approx_model = models.Polynomial1D(
            5,
            c0=-6.54870813,
            c1=-2.15040843,
            c2=-0.28993623,
            c3=-0.04099376,
            c4=-0.00046837,
            c5=-0.00022305,
        ) + models.Exponential1D(amplitude=7.48787292, tau=2.6876055)
        return approx_model(n)

    @staticmethod
    def p0502_to_epsilon(p0502):
        """Petrosian concentration index P0502 (uncorrected quantity) to epsilon"""
        approx_model = models.Polynomial1D(
            6,
            c0=1.09339566,
            c1=-0.14524911,
            c2=0.50361697,
            c3=-0.1215809,
            c4=0.02533795,
            c5=-0.00196243,
            c6=0.00009081,
        ) + models.Exponential1D(amplitude=0.03312881, tau=1.83616642)
        return approx_model(p0502)
