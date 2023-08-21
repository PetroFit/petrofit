import numpy as np

from scipy.interpolate import interp1d

from matplotlib import pyplot as plt

from ..utils import closest_value_index, get_interpolated_values, pixel_to_angular, mpl_tick_frame
from ..photometry import radial_elliptical_aperture

__all__ = [
    'calculate_petrosian', 'calculate_petrosian_r',
    'calculate_concentration_index', 'Petrosian', 'fraction_to_r',
]


def calculate_petrosian(area_list, flux_list, area_err=None, flux_err=None):
    """
    Given a list of aperture areas and associated fluxes, and their errors,
    computes the petrosian curve and its 1-sigma errors.

    Parameters
    ----------
    area_list : numpy.array
        Array of aperture areas.

    flux_list : numpy.array
        Array of photometric flux values.

    area_err : numpy.array
        Array of errors in the aperture areas. If None and `flux_err` is provided,
        errors are calculated with `area_err` is set to zero.

    flux_err : numpy.array
        Array of errors in the flux values. If None and `area_err` is provided,
        errors are calculated with `flux_err` is set to zero.

    Returns
    -------
    petrosian_list : numpy.array
        Array of petrosian values at each value of `area_list`

    petrosian_err : numpy.array
        Array of 1-sigma errors in the Petrosian values.
    """

    # Convert input to array
    area_list = np.array(area_list)
    flux_list = np.array(flux_list)

    # Validate input
    assert len(area_list) == len(flux_list)
    assert len(area_list) > 2, "At least 3 data points are needed to compute Petrosian."
    assert np.all(np.diff(area_list) >= 0)

    # Compute the difference between consecutive elements in the area and flux arrays
    area_diff = np.diff(area_list)
    flux_diff = np.diff(flux_list)

    # Compute the average surface brightness within each aperture
    I_avg_within_r = flux_list[1:] / area_list[1:]

    # Compute the surface brightness at the edge of each aperture
    zero_mask = np.where(area_diff > 0)
    I_at_r = np.zeros_like(area_diff)
    I_at_r[zero_mask] = flux_diff[zero_mask] / area_diff[zero_mask]

    # Compute the Petrosian value at each radius
    petrosian_list = I_at_r / I_avg_within_r

    # Append the first Petrosian value to the beginning of the array
    petrosian_list = np.insert(petrosian_list, 0, np.nan)

    # Match input
    _, unique_indices = np.unique(area_list, return_inverse=True)
    petrosian_list = petrosian_list[unique_indices]

    if area_err is not None or flux_err is not None:
        # Convert input to array
        area_err = np.zeros_like(area_list) if area_err is None else np.array(area_err)
        flux_err = np.zeros_like(flux_list) if flux_err is None else np.array(flux_err)

        # Validate input
        assert len(area_err) == len(area_list)
        assert len(flux_err) == len(flux_list)

        # Compute the errors in the area and flux differences
        area_diff_err = np.sqrt(area_err[:-1] ** 2 + area_err[1:] ** 2)
        flux_diff_err = np.sqrt(flux_err[:-1] ** 2 + flux_err[1:] ** 2)

        # Compute the error in the surface brightness at the edge of each aperture
        I_at_r_err = np.zeros_like(area_diff)
        I_at_r_err[zero_mask] = abs(I_at_r[zero_mask]) * np.sqrt(
            (flux_diff_err[zero_mask] / flux_diff[zero_mask]) ** 2\
            + (area_diff_err[zero_mask] / area_diff[zero_mask]) ** 2)

        # Compute the error in the average surface brightness within each aperture
        I_avg_within_r_err = abs(I_avg_within_r) * np.sqrt(
            (flux_err[1:] / flux_list[1:]) ** 2 + (area_err[1:] / area_list[1:]) ** 2)

        # Compute the error in the Petrosian value at each radius
        petrosian_err = np.zeros_like(area_diff)
        petrosian_err[zero_mask] = abs(petrosian_list[1:][zero_mask]) * np.sqrt(
            (I_at_r_err[zero_mask] / I_at_r[zero_mask]) ** 2 + \
            (I_avg_within_r_err[zero_mask] / I_avg_within_r[zero_mask]) ** 2)

        # Append the first Petrosian error to the beginning of the array
        petrosian_err = np.insert(petrosian_err, 0, np.nan)

        # Match input
        petrosian_err = petrosian_err[unique_indices]

        return petrosian_list, petrosian_err
    return petrosian_list, None


def calculate_petrosian_r(r_list, petrosian_list, petrosian_err=None, eta=0.2,
                          interp_kind='cubic', interp_num=5000):
    """
    Calculate petrosian radius from photometric values using interpolation.
    The Petrosian radius is defined as the radius at which the petrosian profile equals eta.

    Parameters
    ----------
    r_list : numpy.array
        Array of radii in pixels.

    petrosian_list : numpy.array
        Array of petrosian values at each value of `area_list`

    petrosian_err : numpy.array
        Array of 1-sigma errors in the Petrosian values.

    eta : float, default=0.2
        Eta is the petrosian value which defines the `r_petrosian`.

    interp_kind : str or int, optional
        Specifies the kind of interpolation as a string or as an integer specifying the
        order of the spline interpolator to use. If set to `None`, the radius is computed descretely.
        The string has to be one of ‘linear’, ‘nearest’, ‘nearest-up’, ‘zero’, ‘slinear’, ‘quadratic’,
        ‘cubic’, ‘previous’, or ‘next’. ‘zero’, ‘slinear’, ‘quadratic’ and ‘cubic’ refer to a spline
        interpolation of zeroth, first, second or third order; ‘previous’ and ‘next’ simply return the
        previous or next value of the point; ‘nearest-up’ and ‘nearest’ differ when interpolating half-integers
        (e.g. 0.5, 1.5) in that ‘nearest-up’ rounds up and ‘nearest’ rounds down. Default is ‘linear’.

    interp_num : int
        Number of interpolation function sampling radii.

    Returns
    -------
    r_petro : float or numpy.nan
        Petrosian radius

    r_petro_err : float or numpy.nan
        1-sigma error in r_petro. Computed if petrosian_err is provided.
    """
    # Convert input to array
    r_list = np.array(r_list)
    petrosian_list = np.array(petrosian_list)
    if petrosian_err is not None:
        petrosian_err = np.array(petrosian_err)

    # replace the first element with r=0 and petrosian=1
    # This is because the petrosian for the first radius in r_list can not be
    # computed, we instead replace that value with r=0, since that value is known to equal 0.
    r_list[0] = 0
    petrosian_list[0] = 1

    # Interpolate values
    r_list_new, petrosian_list_new = get_interpolated_values(r_list, petrosian_list,
                                                             kind=interp_kind, num=interp_num)

    idx = closest_value_index(eta, petrosian_list_new)

    r_petro = np.nan if idx is None else r_list_new[idx]

    if petrosian_err is None or np.isnan(r_petro):
        return r_petro, np.nan

    # Compute Errors
    # --------------
    # Convert input to array
    petrosian_err[0] = 0

    # Compute upper 1-sigma r_petro
    r_list_upper, petrosian_list_upper = get_interpolated_values(r_list, petrosian_list + petrosian_err,
                                                                 kind=interp_kind, num=interp_num)
    idx = closest_value_index(eta, petrosian_list_upper)
    r_petro_upper = np.nan if idx is None else r_list_upper[idx]

    # Compute lower 1-sigma r_petro
    r_list_lower, petrosian_list_lower = get_interpolated_values(r_list, petrosian_list - petrosian_err,
                                                                 kind=interp_kind, num=interp_num)
    idx = closest_value_index(eta, petrosian_list_lower)
    r_petro_lower = np.nan if idx is None else r_list_lower[idx]

    # Estimate error
    r_petro_err = (r_petro_upper - r_petro_lower) / 2.

    return r_petro, r_petro_err


def fraction_to_r(fraction, r_list, flux_list, r_petrosian,
                  flux_err=None, r_petrosian_err=None,
                  epsilon=2., epsilon_fraction=0.99,
                  interp_kind='cubic', interp_num=5000):
    """
    Given photometric values and `r_total_flux`, calculate radius which encloses a specified
    fraction of the total flux.

    Parameters
    ----------
    fraction : float
        Fraction of flux total flux enclosed in target radius.

    r_list : numpy.array
        Array of radii in pixels.

    flux_list : numpy.array
        Array of photometric flux values.

    r_petrosian : float
        Petrosian radius

    flux_err : int or numpy.array, optional
        Array of errors in the flux values. If None, erros are not computed even if
        `r_petro_err` is provided.

    r_petrosian_err : float, optional
        1-sigma error in r_petro. If set to `None` and `flux_err` is provided,
        `r_petro_err` is assumed to be 0 when computing errors.

    epsilon : float, default=2.
            Epsilon value (used to determine `r_epsilon`).
            N.B: `r_epsilon = r_petrosian` * epsilon`

    epsilon_fraction: float, default=0.99
        Fraction of total flux that is recovered by `r_petrosian * epsilon`.

    interp_kind : str or int, optional
        Specifies the kind of interpolation used on the curve of growth (i.e `r_list` vs `flux_list`)

    interp_num : int
        Number of interpolation function sampling radii.

    Returns
    -------
    r_fraction_flux : float or numpy.nan
        Radius which encloses a specified fraction of the total flux (determined by the Petrosian profile).

    r_fraction_flux_error : float or numpy.nan
        Error in `r_fraction_flux`.
    """

    r_list = np.array(r_list)
    flux_list = np.array(flux_list)
    if flux_err is not None:
        flux_err = np.array(flux_err)
    if r_petrosian_err is not None:
        r_petrosian_err = np.array(r_petrosian_err)

    r_epsilon = r_petrosian * epsilon

    if r_epsilon > max(r_list):
        return np.nan, np.nan

    f = interp1d(r_list, flux_list, kind='cubic' if interp_kind is None else interp_kind)

    # Flux in r_epsilon
    epsilon_flux = f(r_epsilon)

    # Flux value corrsponding to fraction
    fractional_flux = epsilon_flux * (fraction / epsilon_fraction)

    # Get interp flux list and find r_frac:
    r_list_new, flux_list_new = get_interpolated_values(r_list, flux_list, kind=interp_kind, num=interp_num)
    idx = closest_value_index(fractional_flux, flux_list_new, growing=True)
    r_frac = np.nan if idx is None else r_list_new[idx]

    if np.isnan(r_frac) or flux_err is None:
        return r_frac, np.nan

    # Compute Errors
    # --------------
    r_petrosian_err = 0 if r_petrosian_err is None or np.isnan(r_petrosian_err) else r_petrosian_err
    r_epsilon_err = r_petrosian_err

    # Compute the fractional_flux_err error
    f_lower = interp1d(r_list, flux_list - flux_err, kind='cubic' if interp_kind is None else interp_kind)
    f_upper = interp1d(r_list, flux_list + flux_err, kind='cubic' if interp_kind is None else interp_kind)

    fractional_flux_err_lower = f_lower(r_epsilon - r_epsilon_err)
    fractional_flux_err_upper = f_upper(r_epsilon + r_epsilon_err)

    fractional_flux_err = (fractional_flux_err_upper - fractional_flux_err_lower) / 2.

    # Find r_frac in the range of fractional_flux +/- fractional_flux_err
    fractional_flux_upper = fractional_flux + fractional_flux_err
    fractional_flux_lower = fractional_flux - fractional_flux_err

    r_list_lower, flux_list_lower = get_interpolated_values(r_list, flux_list - flux_err)
    idx = closest_value_index(fractional_flux_upper, flux_list_lower, growing=True)
    r_frac_upper = np.nan if idx is None else r_list_lower[idx]

    r_list_upper, flux_list_upper = get_interpolated_values(r_list, flux_list + flux_err)
    idx = closest_value_index(fractional_flux_lower, flux_list_upper, growing=True)
    r_frac_lower = np.nan if idx is None else r_list_upper[idx]

    r_frac_err = (r_frac_upper - r_frac_lower) / 2.

    return r_frac, r_frac_err


def calculate_concentration_index(fraction_1, fraction_2,
                                  r_list, flux_list, r_petrosian,
                                  flux_err=None, r_petrosian_err=None,
                                  epsilon=2., epsilon_fraction=0.99,
                                  interp_kind='cubic', interp_num=5000):
    """
    Calculates Petrosian concentration index.

    ``concentration_index = C_1_2 = 5 * np.log10( r(fraction_2) / r(fraction_1) )``

    Parameters
    ----------
    fraction_1 : float
        Fraction of total light enclosed by the radius in the denominator.

    fraction_2 : float
        Fraction of total light enclosed by the radius in the numerator.

    r_list : numpy.array
        Array of radii in pixels.

    flux_list : numpy.array
        Array of photometric flux values.

    r_petrosian : float
        Petrosian radius

    flux_err : int or numpy.array, optional
        Array of errors in the flux values. If None, erros are not computed even if
        `r_petro_err` is provided.

    r_petrosian_err : float, optional
        1-sigma error in r_petro. If set to `None` and `flux_err` is provided,
        `r_petro_err` is assumed to be 0 when computing errors.

    epsilon : float, default=2.
            Epsilon value (used to determine `r_epsilon`).
            N.B: `r_epsilon = r_petrosian` * epsilon`

    epsilon_fraction: float, default=0.99
        Fraction of total flux that is recovered by `r_petrosian * epsilon`.

    interp_kind : str or int, optional
        Specifies the kind of interpolation used on the curve of growth (i.e `r_list` vs `flux_list`)

    interp_num : int
        Number of interpolation function sampling radii.

    Returns
    -------
    r_fraction_1, r_fraction_2, concentration_index

        * r_fraction_1 : float or np.nan
            Radius containing `fraction_1` of the total flux.

        * r_fraction_2: float or np.nan
            Radius containing `fraction_2` of the total flux.

        * concentration_index : float or np.nan
            Concentration index
    """

    r1 = fraction_to_r(fraction_1, r_list, flux_list, r_petrosian,
                       flux_err=flux_err, r_petrosian_err=r_petrosian_err,
                       epsilon=epsilon, epsilon_fraction=epsilon_fraction,
                       interp_kind=interp_kind, interp_num=interp_num)

    r2 = fraction_to_r(fraction_2, r_list, flux_list, r_petrosian,
                       flux_err=flux_err, r_petrosian_err=r_petrosian_err,
                       epsilon=epsilon, epsilon_fraction=epsilon_fraction,
                       interp_kind=interp_kind, interp_num=interp_num)

    r1 = r1[0]
    r2 = r2[0]

    if np.any(np.isnan(np.array([r1, r2]))):
        return [np.nan, np.nan, np.nan]

    return r1, r2, 5 * np.log10(r2 / r1)


class Petrosian:
    """
    Class that computes and plots Petrosian properties.
    """
    _eta = None
    _epsilon = None
    _epsilon_fraction = None
    _total_flux_fraction = None

    _r_plot_alpha = 0.7

    def __init__(self, r_list, area_list, flux_list, area_err=None, flux_err=None,
                 eta=0.2, epsilon=2., epsilon_fraction=0.99, total_flux_fraction=0.99,
                 verbose=False, interp_kind='cubic', interp_num=5000):
        """
        Petrosian properties.

        Parameters
        ----------
        r_list : numpy.array
            Array of radii in pixels.

        area_list : numpy.array
            Array of aperture areas.

        flux_list : numpy.array
            Array of photometric flux values.

        area_err : numpy.array
            Array of errors in the aperture areas. If None and `flux_err` is provided,
            errors are calculated with `area_err` set to zero.

        flux_err : numpy.array
            Array of errors in the flux values. If None and `area_err` is provided,
            errors are calculated with `flux_err` is set to zero.

        eta : float, default=0.2
            Eta is the petrosian value which defines the `r_petrosian`.

        epsilon : float
            Epsilon value (used to determine `r_total_flux`).
            N.B: `r_total_flux` = `r_petrosian` * `epsilon`

        epsilon_fraction: float, default=0.99
            Fraction of total flux recovered by `r_petrosian * epsilon`.

        total_flux_fraction : float, default=0.99
            Fraction of Sersic flux that defines the Petrosian total flux. Sersic total flux
            is the flux at infinity, thus a smaller fraction must be used to define the total flux
            when analysing images. `total_flux_fraction` can also be adjusted if the image has low
            signal-to-noise or if the profile extends too far out (for example profiles with high Sersic indices).

        verbose : bool
            Prints info

        interp_kind : str or int, optional
            Specifies the kind of interpolation used on the curve of growth (i.e `r_list` vs `flux_list`)

        interp_num : int
            Number of interpolation function sampling radii.
        """

        self.verbose = verbose

        self._r_list = np.array(r_list)
        self._area_list = np.array(area_list)
        self._flux_list = np.array(flux_list)

        self._area_err = None if area_err is None else np.array(area_err)
        self._flux_err = None if flux_err is None else np.array(flux_err)

        self._validate_input_arrays()

        self.epsilon = float(epsilon)
        self.eta = float(eta)
        self.epsilon_fraction = float(epsilon_fraction)
        self.total_flux_fraction = float(total_flux_fraction)

        self.petrosian_list = None
        self.petrosian_err = None
        self.has_petrosian_err = None

        self.interp_kind = interp_kind
        self.interp_num = interp_num

        self._update_petrosian()

    def _validate_input_arrays(self):
        """
        Check if input arrays are equal in size and have at least 3 data points
        """
        assert len(self.r_list) > 2, "At least 3 data points are needed to compute Petrosian."
        assert len(self.r_list.shape) == 1, "Input arrays should be one dimensional."
        assert self.r_list.shape == self.flux_list.shape

        if self.flux_err is not None:
            assert self.flux_list.shape == self.flux_err.shape
        if self.area_err is not None:
            assert self.area_list.shape == self.area_err.shape

    def _update_petrosian(self):
        """
        Updates the Petrosian properties based on current parameters.
        """
        self.petrosian_list, self.petrosian_err = calculate_petrosian(self.area_list, self.flux_list,
                                                                      area_err=self.area_err,
                                                                      flux_err=self.flux_err)
        self.has_petrosian_err = self.petrosian_err is not None

    def _calculate_petrosian_r(self):
        """
        Calculates the Petrosian radius based on the current Petrosian profile.
        """
        return calculate_petrosian_r(self.r_list, self.petrosian_list,
                                     petrosian_err=self.petrosian_err,
                                     eta=self.eta,
                                     interp_kind=self.interp_kind,
                                     interp_num=self.interp_num)

    def _calculate_fraction_to_r(self, fraction):
        """
        Calculates the radius containing a given fraction of the total flux.
        """
        return fraction_to_r(fraction, self.r_list, self.flux_list, self.r_petrosian,
                             flux_err=self.flux_err, r_petrosian_err=self.r_petrosian_err,
                             epsilon=self.epsilon, epsilon_fraction=self.epsilon_fraction,
                             interp_kind=self.interp_kind, interp_num=self.interp_num)

    def concentration_index(self, fraction_1=0.2, fraction_2=0.8):
        """
        Calculates Petrosian concentration index.

        ``concentration_index = C_1_2 = 5 * np.log10( r(fraction_2) / r(fraction_1) )``

        Parameters
        ----------
        fraction_1 : float
            Fraction of total light enclosed by the radius in the denominator.

        fraction_2 : float
            Fraction of total light enclosed by the radius in the numerator.

        Returns
        -------
        r_fraction_1, r_fraction_2, concentration_index

            * r_fraction_1 : float or np.nan
                Radius containing `fraction_1` of the total flux.

            * r_fraction_2: float or np.nan
                Radius containing `fraction_2` of the total flux.

            * concentration_index : float or np.nan
                Concentration index
        """
        return calculate_concentration_index(fraction_1, fraction_2,
                                             self.r_list, self.flux_list, self.r_petrosian,
                                             flux_err=self.flux_err, r_petrosian_err=self.r_petrosian_err,
                                             epsilon=self.epsilon, epsilon_fraction=self.epsilon_fraction,
                                             interp_kind=self.interp_kind, interp_num=self.interp_num)

    @property
    def r_list(self):
        return self._r_list

    @property
    def area_list(self):
        return self._area_list

    @property
    def area_err(self):
        return self._area_err

    @property
    def flux_list(self):
        return self._flux_list

    @property
    def flux_err(self):
        return self._flux_err

    @property
    def epsilon(self):
        """
        Epsilon value (used to determine `r_total_flux`).
        N.B: ``r_total_flux = r_petrosian * epsilon``
        """
        return self._epsilon

    @epsilon.setter
    def epsilon(self, value):
        self._epsilon = value

    @property
    def eta(self):
        """Eta is the Petrosian value which defines the `r_petrosian`"""
        return self._eta

    @eta.setter
    def eta(self, value):
        self._eta = value

    @property
    def epsilon_fraction(self):
        """Fraction of total flux recovered by `epsilon`"""
        return self._epsilon_fraction

    @epsilon_fraction.setter
    def epsilon_fraction(self, value):
        self._epsilon_fraction = value

    @property
    def total_flux_fraction(self):
        """
        Fraction of Sersic flux that defines the Petrosian total flux. Sersic total flux
        is the flux at infinity, thus a smaller fraction must be used to define the total flux
        when analysing images. `total_flux_fraction` can also be adjusted if the image has low
        signal-to-noise or if the profile extends too far out (for example profiles with high Sersic indices).
        """
        return self._total_flux_fraction

    @total_flux_fraction.setter
    def total_flux_fraction(self, value):
        self._total_flux_fraction = value

    @property
    def r_petrosian(self):
        """
        The Petrosian radius is defined as the radius at which the Petrosian profile equals eta.
        """
        r_p, _ = self._calculate_petrosian_r()
        return r_p

    @property
    def r_petrosian_err(self):
        """Estimated 1-sigma r_petrosian Error."""
        _, r_p_err = self._calculate_petrosian_r()
        return r_p_err

    @property
    def r_epsilon(self):
        """r_epsilon = r_petrosian * epsilon"""
        return self.r_petrosian * self.epsilon

    @property
    def r_epsilon_in_range(self):
        """True if `r_epsilon` is in input radii list"""
        return self.r_list.min() <= self.r_epsilon <= self.r_list.max()

    @property
    def r_total_flux(self):
        """The radius containing the total flux"""
        r_frac, r_frac_err = self._calculate_fraction_to_r(self.total_flux_fraction)
        return r_frac

    @property
    def r_total_flux_err(self):
        """The radius containing the total flux"""
        r_frac, r_frac_err = self._calculate_fraction_to_r(self.total_flux_fraction)
        return r_frac_err

    @property
    def total_flux(self):
        """Returns the flux at `r_total_flux` or np.nan if out of range"""
        r_total_flux = self.r_total_flux

        if not min(self.r_list) <= r_total_flux <= max(self.r_list):
            return np.nan

        f = interp1d(self.r_list, self.flux_list, kind=self.interp_kind)
        total_flux = f(r_total_flux)
        return float(total_flux)

    @property
    def total_flux_err(self):
        """
        Returns the photometric uncertainty at `r_total_flux` or np.nan if out of range.
        Note that this includes errors in estimating `r_total_flux`.
        """
        r_total_flux = self.r_total_flux
        r_total_flux_err = self.r_total_flux_err

        if self.flux_err is None or np.isnan(self.total_flux):
            return np.nan

        f = interp1d(self.r_list, self.flux_err, kind='nearest')
        flux_err = f(self.r_total_flux)
        return np.sqrt(flux_err ** 2 + r_total_flux_err ** 2)

    @property
    def half_flux(self):
        """Returns the flux at `r_half_flux` or np.nan if out of range"""
        r_half_flux = self.r_half_light

        if not min(self.r_list) <= r_half_flux <= max(self.r_list):
            return np.nan

        f = interp1d(self.r_list, self.flux_list, kind=self.interp_kind)
        total_flux = f(r_half_flux)
        return float(total_flux)

    @property
    def half_flux_err(self):
        """
        Returns the photometric uncertainty at `r_half_flux` or np.nan if out of range.
        Note that this includes errors in estimating `r_half_flux`.
        """
        r_half_flux_err = self.r_half_flux_err

        if self.flux_err is None or np.isnan(self.half_flux):
            return np.nan

        f = interp1d(self.r_list, self.flux_err, kind='nearest')
        flux_err = f(self.r_half_flux)
        return np.sqrt(flux_err ** 2 + r_half_flux_err ** 2)

    @property
    def r_half_light(self):
        """R_e or Radius containing half of the total Petrosian flux"""
        r_e, r_e_err = self._calculate_fraction_to_r(0.5)
        return r_e

    @property
    def r_half_light_err(self):
        """Error estimate on r_e"""
        r_e, r_e_err = self._calculate_fraction_to_r(0.5)
        return r_e_err

    # Alias for r_half_light
    @property
    def r_50(self):
        return self.r_half_light

    @property
    def r_50_err(self):
        return self.r_half_light_err

    def fraction_flux_to_r(self, fraction=0.5):
        """Given a fraction, compute the radius containing that fraction of the Petrosian total flux"""
        r_frac, r_frac_err = self._calculate_fraction_to_r(fraction)
        return r_frac

    def fraction_flux_to_r_err(self, fraction=0.5):
        """Given a fraction, compute the radius containing that fraction of the Petrosian total flux"""
        r_frac, r_frac_err = self._calculate_fraction_to_r(fraction)
        return r_frac_err

    def r_half_light_arcsec(self, wcs):
        """
        Given a wcs, compute the radius containing half of the total Petrosian flux in arcsec.
        """
        r_half_light = self.r_half_light
        if not np.isnan(r_half_light):
            return pixel_to_angular(r_half_light, wcs).value
        return np.nan

    def r_total_flux_arcsec(self, wcs):
        """
        Given a wcs, compute the radius containing the total Petrosian flux in arcsec.
        """
        r_total_flux = self.r_total_flux
        if not np.isnan(r_total_flux):
            return pixel_to_angular(r_total_flux, wcs).value
        return np.nan

    @property
    def c2080(self):
        """``c2080 = 5 * np.log10(r_80 / r_20)``"""
        return self.concentration_index(fraction_1=0.2, fraction_2=0.8)[-1]

    @property
    def c5090(self):
        """``c5090 = 5 * np.log10(r_90 / r_50)``"""
        return self.concentration_index(fraction_1=0.5, fraction_2=0.9)[-1]

    def _plot_radii(self, ax, radius_unit='pix'):
        radius_unit = '' if radius_unit is None else str(radius_unit)

        r_petrosian = self.r_petrosian
        if not np.isnan(r_petrosian):
            ax.axvline(r_petrosian, linestyle='--', color='black', alpha=self._r_plot_alpha,
                       label=r"$R_{{p}}(\eta_{{{}}})={:0.4f}$ {}".format(self.eta, r_petrosian, radius_unit))

        r_total_flux = self.r_total_flux
        if not np.isnan(r_total_flux):
            total_flux_fraction = int(self.total_flux_fraction * 100)
            ax.axvline(r_total_flux, linestyle='--', c='tab:red', alpha=self._r_plot_alpha,
                       label="$R_{{total}}(L_{{{}}}) = {:0.4f}$ {}".format(total_flux_fraction, r_total_flux,
                                                                           radius_unit))

        r_half_light = self.r_half_light
        if not np.isnan(r_half_light):
            ax.axvline(r_half_light, linestyle='--', c='tab:blue', alpha=self._r_plot_alpha,
                       label="$R_{{50}}(L_{{50}}) = {:0.4f}$ {}".format(r_half_light, radius_unit))

    def plot(self, plot_r=True, title='Petrosian Profile',
             radius_unit='pix', ax=None, color='tab:blue',
             err_alpha=0.2, err_capsize=3,
             show_legend=True, legend_fontsize=None,
             ax_fontsize=None, tick_fontsize=None):
        """
        Plots the Petrosian profile.

        Parameters
        ----------
        plot_r : bool, optional
            If True, plots the total flux and half-light radii. Default is True.

        title : str, optional
            Title for the plot. Default is 'Petrosian Profile'.

        radius_unit : str, optional
            Unit for the radius. Default is 'pix'.

        ax : matplotlib.axis, optional
            Matplotlib axis object to plot on. If None, creates a new axis.

        color : string
            Matplotlib color for profile.

        err_alpha : float, optional
            Transparency for the error region. Default is 0.2.

        err_capsize : int, optional
            Cap size for the error bars. Default is 3.

        show_legend : bool, optional
            If True, displays the legend. Default is True.

        legend_fontsize : int or float, optional
            Font size for the legend. If None, uses default font size.

        ax_fontsize : int or float, optional
            Font size for the axis labels. If None, uses default font size.

        tick_fontsize : int or float, optional
            Font size for the tick labels. If None, uses default font size.

        Returns
        -------
        ax : matplotlib.axis
            Matplotlib axis object with the plot.
        """
        radius_unit = '' if radius_unit is None else str(radius_unit)
        if ax is None:
            ax = plt.gca()

        ax.errorbar(self.r_list, self.petrosian_list, yerr=self.petrosian_err,
                    marker='o', capsize=err_capsize,
                    label="Data", color=color)

        if err_alpha is not None and self.has_petrosian_err and err_alpha > 0:
            ax.fill_between(self.r_list,
                            self.petrosian_list - self.petrosian_err,
                            self.petrosian_list + self.petrosian_err,
                            alpha=err_alpha, color=color)

        r_petrosian = self.r_petrosian
        r_petrosian_err = self.r_petrosian_err

        if plot_r:
            r_color = 'black'
            ax.axhline(self.eta, linestyle='--', color=r_color, alpha=self._r_plot_alpha)
            if not np.isnan(r_petrosian):
                ax.axvline(r_petrosian, linestyle='--', color=r_color, alpha=self._r_plot_alpha,
                           label=r"$R_{{p}}(\eta_{{{}}})={:0.4f}$ {}".format(self.eta, r_petrosian, radius_unit))
                if not np.isnan(r_petrosian_err):
                    ax.errorbar(r_petrosian, self.eta, xerr=r_petrosian_err, zorder=6,
                                marker='o', capsize=5, lw=3, color='tab:orange')
                else:
                    ax.scatter(r_petrosian, self.eta, zorder=6, marker='o', color='tab:orange')

        ax.axhline(0, c='black')
        ax.set_title(title, fontsize=ax_fontsize)
        ax.set_xlabel("Aperture Radius" + " [{}]".format(radius_unit) if radius_unit else "", fontsize=ax_fontsize)
        ax.set_ylabel(r"Petrosian Index $\eta(r)$", fontsize=ax_fontsize)

        mpl_tick_frame(minorticks=True, tick_fontsize=tick_fontsize)

        ax.set_xlim(0, None)
        if show_legend:
            ax.legend(fontsize=legend_fontsize)

        return ax

    def plot_cog(self, plot_r=True, title='Curve of Growth',
                 radius_unit='pix', flux_unit='',
                 ax=None, color='tab:blue',
                 err_alpha=0.2, err_capsize=3,
                 show_legend=True, legend_fontsize=None,
                 ax_fontsize=None, tick_fontsize=None):
        """
        Plots the Curve of Growth (COG) for the Petrosian profile.

        Parameters
        ----------
        plot_r : bool, optional
            If True, plots radii of interest including Petrosian radius.
            Default is True.

        title : str, optional
            Title for the plot. Default is 'Curve of Growth'.

        radius_unit : str, optional
            Unit for the radius. Default is 'pix'.

        flux_unit  : str, optional
            Unit for the cumulative flux. Default is ''

        ax : matplotlib.axis, optional
            Matplotlib axis object to plot on. If None, creates a new axis.

        color : string
            Matplotlib color for profile.

        err_alpha : float, optional
            Transparency for the error region. Default is 0.2.

        err_capsize : int, optional
            Cap size for the error bars. Default is 3.

        show_legend : bool, optional
            If True, displays the legend. Default is True.

        legend_fontsize : int or float, optional
            Font size for the legend. If None, uses default font size.

        ax_fontsize : int or float, optional
            Font size for the axis labels. If None, uses default font size.

        tick_fontsize : int or float, optional
            Font size for the tick labels. If None, uses default font size.

        Returns
        -------
        ax : matplotlib.axis
            Matplotlib axis object with the plot.
        """
        radius_unit = '' if radius_unit is None else str(radius_unit)
        if ax is None:
            ax = plt.gca()

        ax.errorbar(self.r_list, self.flux_list, yerr=self.flux_err,
                    marker='o', capsize=err_capsize, c=color,
                    label="Data")

        if err_alpha is not None and self.has_petrosian_err and err_alpha > 0:
            ax.fill_between(self.r_list,
                            self.flux_list - self.flux_err,
                            self.flux_list + self.flux_err,
                            alpha=err_alpha, color=color)

        if plot_r:
            r_half_light = self.r_half_light
            half_flux = self.half_flux
            if not np.isnan(r_half_light) and not np.isnan(half_flux):
                ax.axvline(r_half_light, linestyle='--', c='black', alpha=self._r_plot_alpha,
                           label="$R_{{50}}(L_{{50}}) = {:0.4f}$ {}".format(r_half_light, radius_unit))
                ax.axhline(half_flux, linestyle='--', c='black', alpha=self._r_plot_alpha)
                ax.scatter(r_half_light, half_flux, zorder=6, marker='o', color='tab:orange')

            r_total_flux = self.r_total_flux
            total_flux = self.total_flux
            if not np.isnan(r_total_flux) and not np.isnan(total_flux):
                total_flux_fraction = int(self.total_flux_fraction * 100)
                ax.axvline(r_total_flux, linestyle='-', c='black', alpha=self._r_plot_alpha,
                           label="$R_{{total}}(L_{{{}}}) = {:0.4f}$ {}".format(total_flux_fraction, r_total_flux,
                                                                               radius_unit))
                ax.axhline(total_flux, linestyle='-', c='black', alpha=self._r_plot_alpha)
                ax.scatter(r_total_flux, total_flux, zorder=6, marker='o', color='tab:orange')

        ax.set_title(title, fontsize=ax_fontsize)
        ax.set_xlabel("Aperture Radius" + " [{}]".format(radius_unit) if radius_unit else "", fontsize=ax_fontsize)
        ax.set_ylabel(r"$L(\leq r)$" + (" [{}]".format(flux_unit) if flux_unit else ''), fontsize=ax_fontsize)

        mpl_tick_frame(minorticks=True, tick_fontsize=tick_fontsize)

        ax.set_xlim(0, None)
        ax.set_ylim(0, None)
        if show_legend:
            ax.legend(fontsize=legend_fontsize)

        return ax

    def imshow(self, position=(0, 0), elong=1., theta=0., color=None, lw=None):
        """
        Make 2D plots of elliptical apertures with radii  of `r_half_light`, `r_total_flux`, `r_20` and `r_80`.

        Parameters
        ----------
        position : tuple
            (x, y) center of the apertures.

        elong : float
            Elongation of the aperture.

        theta : float
            The orientation of the aperture in rad.

        color : str
            Color override that will change the color of the apertures in the plot.

        lw : float
            Line width (thickness) of the plotted apertures.
        """

        labels = ['r_half_light', 'r_total_flux', 'r_20', 'r_80']
        radii = [self.r_half_light, self.r_total_flux, self._calculate_fraction_to_r(.2)[0],
                 self._calculate_fraction_to_r(.8)[0]]
        colors = ['r', 'r', 'b', 'b']
        linestyles = ['dashed', 'solid', 'dotted', 'dashdot']
        for label, r, default_color, ls in zip(labels, radii, colors, linestyles):
            if not np.isnan(r) and r > 0:
                radial_elliptical_aperture(position, r, elong, theta).plot(
                    label=label, linestyle=ls, color=color if color else default_color, lw=lw)

        plt.scatter(*position, marker='+', color=color if color else 'red')
