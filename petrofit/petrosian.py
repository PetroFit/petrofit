import yaml

import numpy as np

from scipy.interpolate import interp1d

from matplotlib import pyplot as plt

from sklearn.tree import DecisionTreeRegressor

from .utils import closest_value_index, get_interpolated_values, pixel_to_angular


def plot_petrosian(r_list, area_list, flux_list, epsilon=2., eta=0.2, plot_r=False):
    """
    Given photometric values, plots petrosian profile.

    Parameters
    ----------
    r_list : numpy.array
        Array of radii in pixels.

    area_list : numpy.array
        Array of aperture areas.

    flux_list : numpy.array
        Array of photometric flux values.

    epsilon : float
        Epsilon value (used to determine `r_total_flux`).
        N.B: `r_total_flux` = `r_petrosian` * `epsilon`

    eta : float, default=0.2
        Eta is the petrosian value which defines the `r_petrosian`.

    plot_r : bool
        If set to True, `r_half_light` and `r_total_flux` will be plotted.
    """
    petrosian_list = calculate_petrosian(area_list, flux_list)

    r_list_new, petrosian_list_new = get_interpolated_values(r_list, petrosian_list)

    plt.plot(r_list, petrosian_list, marker='o', linestyle='None', label='Data')
    plt.plot(r_list_new, petrosian_list_new, label='Interpolated [cubic]')

    r_petrosian = calculate_petrosian_r(r_list, area_list, flux_list, eta=eta)
    if not np.isnan(r_petrosian):
        plt.axvline(r_petrosian, linestyle='--', label="r_petrosian={:0.4f} pix".format(r_petrosian))
        plt.axhline(eta, linestyle='--', label='eta={:0.4f}'.format(eta))


    r_total = calculate_r_total_flux(r_list, area_list, flux_list, epsilon, eta)
    if not np.isnan(r_total) and plot_r:
        plt.axvline(r_total, linestyle='--', label="r_total={:0.4f} pix".format(r_total), c='black')

    r_half_light = calculate_r_half_light(r_list, flux_list, r_total)
    if not np.isnan(r_half_light) and plot_r:
        plt.axvline(r_half_light, linestyle='--', label="r_half_light={:0.4f} pix".format(r_half_light), c='gray')

    plt.legend(loc='best')

    plt.title("Petrosian")
    plt.xlabel("Aperture Radius [Pix]")
    plt.ylabel("Petrosian Value")


def calculate_petrosian(area_list, flux_list):
    """
    Given a list of aperture areas and associated fluxes, computes the petrosian curve.

    Parameters
    ----------
    area_list : numpy.array
        Array of aperture areas.

    flux_list : numpy.array
        Array of photometric flux values.

    Returns
    -------
    petrosian_list : numpy.array
        Array of petrosian values at each value of `area_list`
    """

    petrosian_list = []

    last_area = 0
    last_I = 0
    petrosian_value = 0
    for i in range(len(area_list)):
        area = area_list[i]
        I = flux_list[i]

        if area != last_area:
            area_of_slice = area - last_area
            I_at_r = (I - last_I) / area_of_slice

            area_within_r = area
            I_avg_within_r = (I / area_within_r)

            petrosian_value = I_at_r / I_avg_within_r

        petrosian_list.append(petrosian_value)

        last_area = area
        last_I = I

    return np.array(petrosian_list)


def calculate_petrosian_r(r_list, area_list, flux_list, eta=0.2):
    """
    Calculate petrosian radius from photometric values using interpolation.
    The petrosian radius is defined as the radius at which the petrosian profile equals eta.

    Parameters
    ----------
    r_list : numpy.array
        Array of radii in pixels.

    area_list : numpy.array
        Array of aperture areas.

    flux_list : numpy.array
        Array of photometric flux values.

    eta : float, default=0.2
        Eta is the petrosian value which defines the `r_petrosian`.

    Returns
    -------
    r_petrosian : float or numpy.nan
        Petrosian radius
    """
    petrosian_list = calculate_petrosian(area_list, flux_list)

    r_list_new, petrosian_list_new = get_interpolated_values(r_list, petrosian_list)

    idx = closest_value_index(eta, petrosian_list_new)

    return np.nan if idx is None else r_list_new[idx]


def discrete_petrosian_r(r_list, area_list, flux_list, eta=0.2):
    """
    Calculate petrosian radius from photometric values by using the nearest value petrosian to eta.
    The petrosian radius is defined as the radius at which the petrosian profile equals eta.

    Parameters
    ----------
    r_list : numpy.array
        Array of radii in pixels.

    area_list : numpy.array
        Array of aperture areas.

    flux_list : numpy.array
        Array of photometric flux values.

    eta : float, default=0.2
        Eta is the petrosian value which defines the `r_petrosian`.

    Returns
    -------
    r_petrosian : float or numpy.nan
        Petrosian radius
    """

    petrosian_list = calculate_petrosian(area_list, flux_list)
    idx_list = np.where(petrosian_list <= eta)[0]

    r_petrosian = np.nan
    if idx_list.size > 0:
        idx = idx_list[0]
        r_petrosian = r_list[idx]

    return r_petrosian


def calculate_r_total_flux(r_list, area_list, flux_list, epsilon=2., eta=0.2, verbose=False):
    """
    Given photometric values, calculate Petrosian `r_total_flux`.

    Parameters
    ----------
    r_list : numpy.array
        Array of radii in pixels.

    area_list : numpy.array
        Array of aperture areas.

    flux_list : numpy.array
        Array of photometric flux values.

    epsilon : float
        Epsilon value (used to determine `r_total_flux`).
        N.B: `r_total_flux` = `r_petrosian` * `epsilon`

    eta : float, default=0.2
        Eta is the petrosian value which defines the `r_petrosian`.

    Returns
    -------
    r_total_flux : float or numpy.nan
        Total flux radius determined by the Petrosian profile.
    """

    r_petrosian = calculate_petrosian_r(r_list, area_list, flux_list, eta=eta)
    if np.isnan(r_petrosian):
        if verbose:
            print("r_petrosian could not be computed")
        return np.nan

    return r_petrosian * epsilon


def fraction_flux_to_r(r_list, flux_list, r_total_flux, fraction=0.5):
    """
    Given photometric values and `r_total_flux`, calculate radius which encloses a specified
    fraction of the total flux.

    Parameters
    ----------
    r_list : numpy.array
        Array of radii in pixels.

    flux_list : numpy.array
        Array of photometric flux values.

    r_total_flux : float
        Total flux radius (hint: `petrofit.petrosian.calculate_r_total_flux` can be used to measure this).

    fraction : float
        Fraction of flux total flux enclosed in target radius.

    Returns
    -------
    r_fraction_flux : float or numpy.nan
        Radius which encloses a specified fraction of the total flux (determined by the Petrosian profile).
    """

    if r_total_flux > max(r_list):
        return np.nan

    f = interp1d(r_list, flux_list, kind='cubic')
    total_flux = f(r_total_flux)
    fractional_flux = total_flux * fraction

    r_list_new, flux_list_new = get_interpolated_values(r_list, flux_list)

    # idx = abs(flux_list_new - fractional_flux).argmin()
    idx = closest_value_index(fractional_flux, flux_list_new, growing=True)
    return np.nan if idx is None else r_list_new[idx]


def calculate_r_half_light(r_list, flux_list, r_total_flux):
    """
    Given photometric values, calculate Petrosian `r_half_light`.

    Parameters
    ----------
    r_list : numpy.array
        Array of radii in pixels.

    flux_list : numpy.array
        Array of photometric flux values.

    r_total_flux : float
        Total flux radius (hint: `petrofit.petrosian.calculate_r_total_flux` can be used to measure this).

    Returns
    -------
    r_half_light : float or numpy.nan
        Radius containing half of total flux.
    """
    return fraction_flux_to_r(r_list, flux_list, r_total_flux, fraction=0.5)


def calculate_concentration_index(r_list, flux_list, r_total_flux, fraction_1=0.2, fraction_2=0.8):
    """
    Calculates Petrosian concentration index.

    ``concentration_index = 5 * np.log10( r(fraction_2) / r(fraction_1) )``

    Parameters
    ----------
    r_list : numpy.array
        Array of radii in pixels.

    flux_list : numpy.array
        Array of photometric flux values.

    r_total_flux : float
        Total flux radius (hint: `petrofit.petrosian.calculate_r_total_flux` can be used to measure this).

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

    if r_total_flux > max(r_list):
        return [np.nan, np.nan, np.nan]

    r1 = fraction_flux_to_r(r_list, flux_list, r_total_flux, fraction=fraction_1)
    r2 = fraction_flux_to_r(r_list, flux_list, r_total_flux, fraction=fraction_2)

    # if np.any(np.isnan(np.array([r1, r2]))):
    #     return [np.nan, np.nan, np.nan]

    return r1, r2, 5 * np.log10(r2 / r1)


def estimate_n(c2080pet, verbose=False):
    n_list = [0.5, 0.75, 1, 1.5, 2, 4, 6, 8]
    c_pet_list = [2.14, 2.49, 2.78, 3.26, 3.63, 4.50, 4.99, 5.31]
    f = interp1d(c_pet_list, n_list, kind='cubic')
    try:
        return f(c2080pet)
    except ValueError:
        if verbose:
            print("Could not estimate n for {}, returning closest".format(c2080pet))
        return 0.5 if c2080pet < 2.14 else 5.31


class Petrosian:
    """
    Class that computes and plots Petrosian properties.
    """

    def __init__(self, r_list, area_list, flux_list,
                 epsilon=2., eta=0.2, verbose=False):
        """
        Initialize Petrosian properties by providing flux information.

        Parameters
        ----------
        r_list : numpy.array
            Array of radii in pixels.

        area_list : numpy.array
            Array of aperture areas.

        flux_list : numpy.array
            Array of photometric flux values.

        epsilon : float
            Epsilon value (used to determine `r_total_flux`).
            N.B: `r_total_flux` = `r_petrosian` * `epsilon`

        eta : float, default=0.2
            Eta is the petrosian value which defines the `r_petrosian`.

        verbose : bool
            Prints info using
        """

        self.verbose = verbose

        self.r_list = r_list
        self.area_list = area_list
        self.flux_list = flux_list

        self._epsilon = None
        self._eta = None

        self.epsilon = epsilon
        self.eta = eta

    @property
    def epsilon(self):
        """
        Epsilon value (used to determine `r_total_flux`).
        N.B: `r_total_flux` = `r_petrosian` * `epsilon
        """
        return self._epsilon

    @epsilon.setter
    def epsilon(self, value):
        self._epsilon = value

    @property
    def eta(self):
        """Eta is the petrosian value which defines the `r_petrosian`"""
        return self._eta

    @eta.setter
    def eta(self, value):
        self._eta = value

    @property
    def r_petrosian(self):
        """
        The Petrosian radius is defined as the radius at which the petrosian profile equals eta.
        """
        return calculate_petrosian_r(self.r_list, self.area_list, self.flux_list,
                                     eta=self.eta)

    @property
    def r_total_flux(self):
        return calculate_r_total_flux(self.r_list, self.area_list, self.flux_list,
                                      epsilon=self.epsilon, eta=self.eta,
                                      verbose=self.verbose)

    @property
    def r_half_light(self):
        """Radius containing half of the total Petrosian flux"""
        return calculate_r_half_light(self.r_list, self.flux_list, self.r_total_flux)

    def r_half_light_arcsec(self, wcs):
        """
        Given a wcs, compute the radius containing half of the total Petrosian flux in arcsec.
        """
        if not np.isnan(self.r_half_light):
            return pixel_to_angular(self.r_half_light, wcs).value
        return np.nan

    def r_total_flux_arcsec(self, wcs):
        """
        Given a wcs, compute the radius containing the total Petrosian flux in arcsec.
        """
        if not np.isnan(self.r_total_flux):
            return pixel_to_angular(self.r_total_flux, wcs).value
        return np.nan

    def concentration_index(self, fraction_1=0.2, fraction_2=0.8):
        """
        Calculates Petrosian concentration index.

        ``concentration_index = 5 * np.log10( r(fraction_2) / r(fraction_1) )``

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
        return calculate_concentration_index(self.r_list, self.flux_list, self.r_total_flux,
                                             fraction_1=fraction_1, fraction_2=fraction_2)

    def fraction_flux_to_r(self, fraction=0.5):
        """Given a fraction, compute the radius containing that fraction of the Petrosian total flux"""
        return self.fraction_flux_to_r(self.r_list, self.flux_list, self.r_total_flux, fraction=fraction)


    def plot(self, plot_r=False, plot_normalized_flux=False):
        """
        Plots Petrosian profile.

        Parameters
        ----------
        plot_r : bool
            Plot total flux and half light radii.

        plot_normalized_flux:
            Over-plot the flux curve of growth by normalize the flux axis (max_flux=1).
        """
        plot_petrosian(self.r_list, self.area_list, self.flux_list, epsilon=self.epsilon, eta=self.eta, plot_r=plot_r)

        if plot_normalized_flux:
            plt.plot(self.r_list, self.flux_list/self.flux_list.max(), label='Normalized Flux', linestyle='--')


class PetrosianCorrection:
    """
    This class is experimental.
    Efforts to provide Petrosian corrections.
    """
    def __init__(self, grid_file):

        self._grid_file = grid_file
        self.grid = self._read_grid_file()
        self.regressor = self._train()

    def _read_grid_file(self):
        with open(self._grid_file, 'r') as infile:
            grid = yaml.load(infile, Loader=yaml.Loader)
        return grid

    def _train(self):
        X = []
        N = []

        for r_eff in self.grid:
            for i in range(len(self.grid[r_eff]['n'])):
                c = self.grid[r_eff]['c_index'][i]
                n = self.grid[r_eff]['n'][i]
                r_p = self.grid[r_eff]['r_petrosian'][i]
                epsilon = self.grid[r_eff]['epsilon'][i]
                r_hl_petrosian = self.grid[r_eff]['r_hl_petrosian'][i]
                X.append(np.array([r_p, c]))
                N.append([n, epsilon])

        X = np.array(X)
        N = np.array(N)

        clf = DecisionTreeRegressor()
        clf.fit(X, N)
        return clf

    def estimate_regressor(self, r_pet, c2080pet, verbose=False):
        """ Retrns n, epsilon"""
        return self.regressor.predict([[r_pet, c2080pet]])[0]

    def estimate_n(self, r_hl_pet, c2080pet, verbose=False):

        r_eff = np.round(r_hl_pet / 5) * 5
        if r_eff == 0:
            r_eff = 5

        N_LIST = np.array(self.grid[r_eff]['n'])
        C_LIST = np.array(self.grid[r_eff]['c_index'])

        u, indices = np.unique(C_LIST, return_index=True)

        n_list = N_LIST[indices]
        c_pet_list = C_LIST[indices]
        f = interp1d(c_pet_list, n_list, kind='cubic')
        try:
            return float(f(c2080pet))
        except ValueError:
            if verbose:
                print("Could not estimate n for {}, returning closest".format(c2080pet))
            return 0.5 if c2080pet < 2.14 else 5.31

    def estimate_epsilon(self, r_hl_pet, c2080pet, verbose=False):

        r_eff = np.round(r_hl_pet / 5) * 5
        if r_eff == 0:
            r_eff = 5

        N_LIST = np.array(self.grid[r_eff]['epsilon'])
        C_LIST = np.array(self.grid[r_eff]['c_index'])

        u, indices = np.unique(C_LIST, return_index=True)
        indices = np.array(indices)

        n_list = N_LIST[indices]
        c_pet_list = C_LIST[indices]
        f = interp1d(c_pet_list, n_list, kind='cubic')
        try:
            return float(f(c2080pet))
        except ValueError:
            if verbose:
                print("Could not estimate n for {}, returning closest".format(c2080pet))
            return 2

