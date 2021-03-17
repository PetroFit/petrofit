import yaml

import numpy as np

from scipy.interpolate import interp1d

from matplotlib import pyplot as plt

from sklearn.tree import DecisionTreeRegressor

from .utils import closest_value_index, get_interpolated_values, pixel_to_angular


def plot_petrosian(r_list, area_list, flux_list, epsilon=2., eta=0.2, plot_r=False):
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
    petrosian_list = calculate_petrosian(area_list, flux_list)

    r_list_new, petrosian_list_new = get_interpolated_values(r_list, petrosian_list)

    idx = closest_value_index(eta, petrosian_list_new)

    return np.nan if idx is None else r_list_new[idx]


def discrete_petrosian_r(r_list, area_list, flux_list, eta=0.2):
    petrosian_list = calculate_petrosian(area_list, flux_list)
    idx_list = np.where(petrosian_list <= eta)[0]

    r_petrosian = np.nan
    if idx_list.size > 0:
        idx = idx_list[0]
        r_petrosian = r_list[idx]

    return r_petrosian


def calculate_r_total_flux(r_list, area_list, flux_list, epsilon=2., eta=0.2, verbose=False):
    r_petrosian = calculate_petrosian_r(r_list, area_list, flux_list, eta=eta)
    if np.isnan(r_petrosian):
        if verbose:
            print("r_petrosian could not be computed")
        return np.nan

    return r_petrosian * epsilon


def fraction_flux_to_r(r_list, flux_list, r_total_flux, fraction=0.5):

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
    return fraction_flux_to_r(r_list, flux_list, r_total_flux, fraction=0.5)


def calculate_concentration_index(r_list, flux_list, r_total_flux, ratio1=0.2, ratio2=0.8):

    if r_total_flux > max(r_list):
        return [np.nan, np.nan, np.nan]

    r1 = fraction_flux_to_r(r_list, flux_list, r_total_flux, fraction=ratio1)
    r2 = fraction_flux_to_r(r_list, flux_list, r_total_flux, fraction=ratio2)

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

    def __init__(self, r_list, flux_list, area_list,
                 epsilon=2., eta=0.2, verbose=False):

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
        return self._epsilon

    @epsilon.setter
    def epsilon(self, value):
        self._epsilon = value

    @property
    def eta(self):
        return self._eta

    @eta.setter
    def eta(self, value):
        self._eta = value

    @property
    def r_petrosian(self):
        return calculate_petrosian_r(self.r_list, self.area_list, self.flux_list,
                                     eta=self.eta)

    @property
    def r_total_flux(self):
        return calculate_r_total_flux(self.r_list, self.area_list, self.flux_list,
                                      epsilon=self.epsilon, eta=self.eta,
                                      verbose=self.verbose)

    @property
    def r_half_light(self):
        return calculate_r_half_light(self.r_list, self.flux_list, self.r_total_flux)

    def r_half_light_arcsec(self, wcs):
        if not np.isnan(self.r_half_light):
            return pixel_to_angular(self.r_half_light, wcs).value
        return np.nan

    def r_total_flux_arcsec(self, wcs):
        if not np.isnan(self.r_total_flux):
            return pixel_to_angular(self.r_total_flux, wcs).value
        return np.nan

    def concentration_index(self, ratio1=0.2, ratio2=0.8):
        return calculate_concentration_index(self.r_list, self.flux_list, self.r_total_flux,
                                             ratio1=ratio1, ratio2=ratio2)

    def fraction_flux_to_r(self, fraction=0.5):
        return self.fraction_flux_to_r(self.r_list, self.flux_list, self.r_total_flux, fraction=fraction)


    def plot(self, plot_r=False, plot_normalized_flux=False):
        plot_petrosian(self.r_list, self.area_list, self.flux_list, epsilon=self.epsilon, eta=self.eta, plot_r=plot_r)

        if plot_normalized_flux:
            plt.plot(self.r_list, self.flux_list/self.flux_list.max(), label='Normalized Flux', linestyle='--')


class PetrosianCorrection:
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

