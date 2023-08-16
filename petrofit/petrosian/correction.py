from copy import copy
import warnings

import numpy as np

from scipy.interpolate import interp1d
from scipy.special import gammaincinv

from astropy.utils.console import ProgressBar
from astropy.modeling import models
from astropy.table import Table

from ..modeling.models import PSFConvolvedModel2D, sersic_enclosed, sersic_enclosed_inv
from ..modeling.fitting import model_to_image
from ..photometry import radial_photometry
from .core import Petrosian, calculate_petrosian_r, calculate_petrosian

from matplotlib import pyplot as plt

__all__ = ['generate_petrosian_sersic_correction', 'PetrosianCorrection']


def _generate_petrosian_correction(args):
    """
    Helper function to compute corrections for a single pair of `r_eff` and `n`.
    `args` should be a list `[r_eff, n, psf, oversample, plot]`. See
    `generate_petrosian_sersic_correction` doctring for more information.
    """
    # Unpack params
    r_eff, n, psf, oversample, plot = args
    amplitude = 100 / np.exp(gammaincinv(2. * n, 0.5))

    # Total flux
    L_total = sersic_enclosed(
        np.inf,
        amplitude=amplitude,
        r_eff=r_eff,
        n=n)
    total_flux = L_total * 0.99

    # Calculate radii
    r_20, r_80, r_total_flux = [sersic_enclosed_inv(
        total_flux * fraction,
        amplitude=amplitude,
        r_eff=r_eff,
        n=n) for fraction in [0.2, 0.8, 1.0]]

    # Make r_list
    max_r = r_total_flux * 3 if n < 2 else r_total_flux * 1.3
    if r_eff < 7 and n < 1:
        max_r = r_total_flux * 50
    if max_r >= 200:
        r_list = [x for x in range(1, 201, 2)]
        r_list += [x for x in range(300, int(max_r) + 100, 100)]
    else:
        r_list = [x for x in range(1, int(max_r) + 2, 2)]
    r_list = np.array(r_list)

    image_size = max(r_list) * 2

    x_0 = image_size // 2
    y_0 = image_size // 2

    # Make Model Image
    # ----------------
    # Define model
    galaxy_model = models.Sersic2D(
        amplitude=amplitude,
        r_eff=r_eff,
        n=n,
        x_0=x_0,
        y_0=y_0,
        ellip=0.,
        theta=0.,
    )

    # Wrap model with PSFConvolvedModel2D
    galaxy_model = PSFConvolvedModel2D(galaxy_model, psf=psf, oversample=oversample)

    # Make galaxy image from PSFConvolvedModel2D
    galaxy_image = model_to_image(galaxy_model, image_size, center=(x_0, y_0))

    # Do photometry on model galaxy image
    flux_list, area_list, err = radial_photometry(galaxy_image, (x_0, y_0), r_list,
                                                  plot=plot,
                                                  vmax=amplitude / 100)

    if plot:
        plt.show()
    # Calculate Photometry and petrosian
    # ----------------------------------
    # Petrosian from Photometry
    p = Petrosian(r_list, area_list, flux_list)
    rc1, rc2, c_index = p.concentration_index()
    if np.any(np.isnan(np.array([rc1, rc2, c_index]))):
        raise Exception("concentration_index cannot be computed (n={}, r_e={})".format(n, r_eff))

    # Compute new r_total_flux
    _, indices = np.unique(flux_list, return_index=True)
    indices = np.array(indices)
    f = interp1d(flux_list[indices], r_list[indices], kind='linear')
    model_r_total_flux = f(total_flux)

    # Compute new r_80
    model_r_80 = f(total_flux * 0.8)

    # Compute corrections
    corrected_epsilon = model_r_total_flux / p.r_petrosian
    corrected_epsilon_80 = model_r_80 / p.r_petrosian

    corrected_p = copy(p)
    corrected_p.epsilon = corrected_epsilon

    # Make output list
    # ----------------
    # Petrosian indices
    petrosian_list = calculate_petrosian(p.area_list, p.flux_list)[0]
    p02, p03, p04, p05 = [calculate_petrosian_r(p.r_list, petrosian_list, eta=i)[0] for i in (0.2, 0.3, 0.4, 0.5)]
    assert np.round(p.r_petrosian, 6) == np.round(p02, 6)

    u_r_eff = p.fraction_flux_to_r(0.5)
    u_r_20 = p.fraction_flux_to_r(0.2)
    u_r_50 = p.fraction_flux_to_r(0.5)
    u_r_80 = p.fraction_flux_to_r(0.8)

    c_r_eff = corrected_p.fraction_flux_to_r(0.5)
    c_r_20 = corrected_p.fraction_flux_to_r(0.2)
    c_r_50 = corrected_p.fraction_flux_to_r(0.5)
    c_r_80 = corrected_p.fraction_flux_to_r(0.8)

    row = [n, r_eff, r_20, r_80, r_total_flux, L_total,
           p02, p03, p04, p05, 5 * np.log10(p02 / p05), 5 * np.log10(p02 / p03),
           p.epsilon, u_r_50 / p.r_petrosian, u_r_80 / p.r_petrosian, u_r_eff, p.r_total_flux, u_r_20, u_r_80, p.c2080, p.c5090,
           corrected_epsilon, c_r_50 / p.r_petrosian, corrected_epsilon_80, c_r_eff, corrected_p.r_total_flux, c_r_20, c_r_80,
           corrected_p.c2080, corrected_p.c5090]

    if plot:
        corrected_p.plot(True, True)
        plt.show()
        print(corrected_epsilon)
        print(r_eff, p.r_half_light, corrected_p.r_half_light)
        print(" ")

    del galaxy_model, galaxy_image
    del flux_list, area_list, err
    del corrected_p, p

    return row


def generate_petrosian_sersic_correction(output_file_name, psf=None, r_eff_list=None, n_list=None,
                                         oversample=('x_0', 'y_0', 10, 50), out_format=None, overwrite=False,
                                         ipython_widget=False, n_cpu=None, plot=False):
    """
    Generate corrections for Petrosian profiles by simulating a galaxy image (single component sersic) and measuring its
    properties. This is done to identify the correct `epsilon` value that, when multiplied with `r_petrosian`, gives
    `r_total_flux`. To achieve this, an image is created from a Sersic model and convolved with a PSF (if provided).
    The Petrosian radii and concentrations are computed using the default `epsilon` = 2. Since the real `r_total_flux`
    of the simulated galaxy is known, the correct `epsilon` can be determined by
    `epsilon = r_petrosian / corrceted_r_total_flux`. The resulting grid is used to map measured properties to the
    correct `epsilon` value. If `output_file_name` is provided, the grid is saved to using an astropy table file which
    is readable by `petrofit.petrosian.PetrosianCorrection`.

    Parameters
    ----------
    output_file_name : str
        Name of output file, must have .yaml or .yml extension.

    psf : numpy.array or None
        2D PSF image to pass to `petrofit.fitting.models.PSFConvolvedModel2D`.

    r_eff_list : list, (optional)
        List of `r_eff` (half light radii) in pixels to evaluate.

    n_list : list, (optional)
        List of Sersic indices to evaluate.

    oversample : int or tuple
        oversampling to pass to `petrofit.fitting.models.PSFConvolvedModel2D`.

    out_format : str, optional
        Format passed to the resulting astropy table when writing to file.

    overwrite : bool, optional
        Overwrite if file exists.

    ipython_widget : bool, optional
        If True, the progress bar will display as an IPython notebook widget.

    n_cpu : bool, int, optional
        If True, use the multiprocessing module to distribute each task to a different
        processor core. If a number greater than 1, then use that number of cores. This
        should be selected taking ram in consideration (since high n and large r_eff
        create large images).

    plot : bool
        Shows plot of photometry and Petrosian. Not available if n_cpu > 1.

    Returns
    -------
    petrosian_grid : Table
        Astropy Table that is readable by `petrofit.petrosian.PetrosianCorrection`
    """

    if r_eff_list is None:
        r_eff_list = np.arange(10, 100 + 5, 5)

    if n_list is None:
        n_list = np.arange(0.5, 4.5 + 0.5, 0.5)

    if psf is not None and psf.sum() != 1:
        warnings.warn("Input PSF not normalized to 1, current sum = {}. This may cause major errors".format(psf.sum()))

    r_eff_list = np.array(r_eff_list)
    n_list = np.round(np.array(n_list), 6)

    # Make list of args for _generate_petrosian_correction
    args = []
    for n_idx, n in enumerate(n_list):
        for r_eff_idx, r_eff in enumerate(r_eff_list):
            args.append([r_eff, n, psf, oversample, plot])

    # Call _generate_petrosian_correction
    # either on one thread on using multiprocessing
    if n_cpu is None or n_cpu == 1:
        with ProgressBar(len(args), ipython_widget=ipython_widget) as bar:
            rows = []
            for arg in args:
                row = _generate_petrosian_correction(arg)
                rows.append(row)
                bar.update()
    else:
        assert plot == False, 'Plotting not available for ncpu > 1'
        step = 50 if len(r_eff_list) * len(n_list) > 500 else 2
        rows = ProgressBar.map(_generate_petrosian_correction, args, multiprocess=n_cpu,
                               ipython_widget=ipython_widget, step=step)

    names = ['n', 'r_eff', 'sersic_r_20', 'sersic_r_80', 'sersic_r_99', 'sersic_L_inf',
             'p02', 'p03', 'p04', 'p05', 'p0502', 'p0302',
             'u_epsilon', 'u_epsilon_50', 'u_epsilon_80', 'u_r_50', 'u_r_99', 'u_r_20', 'u_r_80', 'u_c2080', 'u_c5090',
             'c_epsilon', 'c_epsilon_50', 'c_epsilon_80', 'c_r_50', 'c_r_99', 'c_r_20', 'c_r_80', 'c_c2080', 'c_c5090']
    petrosian_grid = Table(rows=rows, names=names)

    if output_file_name is not None:
        try:
            petrosian_grid.write(output_file_name, format=out_format, overwrite=overwrite)
        except Exception as e:
            print('Could not save to file: {}'.format(e))
            print('You can save the returned table using `petrosian_grid.write`')
    return petrosian_grid


class PetrosianCorrection:
    """
    This class computes corrections for Petrosian given default Petrosian measurements.
    """

    def __init__(self, grid, enforce_range=True):
        """
        Parameters
        ----------
        grid : str
            Correction grid generated by `petrofit.correction.generate_petrosian_sersic_correction`.
            Use `PetrosianCorrection.read(file_path)` to read grid from file.

        enforce_range : bool
            If true, the nearest approximation is returned. If false, an assertion will be applied that makes sure
            that the profiles to be corrected are covered by the correction grid.
        """

        self.enforce_range = enforce_range

        self.eta_keys = {0.2: 'p02', 0.3: 'p03', 0.4: 'p04', 0.5: 'p05'}

        if isinstance(grid, Table):
            self.grid = grid
        elif isinstance(grid, str):
            raise TypeError('Input grid should be an astropy Table use `PetrosianCorrection.read(file_path)`')
        else:
            raise TypeError('Input grid should be an astropy Table')

        self.x = self.grid['p02'].value
        self.y = self.grid['u_r_50'].value
        self.z = self.grid['u_c2080'].value
        self.r = [self.x, self.y, self.z]

        self.weights = np.array([100, 100, 100])

    def _get_xyz_from_p(self, p):
        px_list = (0.2, 0.3, 0.4, 0.5)
        p02, p03, p04, p05 = [calculate_petrosian_r(p.r_list, p.petrosian_list,
                                                    petrosian_err=None, eta=i)[0] for i in px_list]
        x0 = p02
        y0 = p.r_50
        z0 = p.c2080
        return [x0, y0, z0]

    @staticmethod
    def _read_grid_file(grid_file, file_format=None):
        return Table.read(grid_file, format=file_format)

    @classmethod
    def read(cls, grid_file, file_format=None):
        """Read grid from file."""
        grid = cls._read_grid_file(grid_file, file_format)
        return cls(grid)

    def write(self, grid_file, file_format=None):
        """Write grid to file."""
        self.grid.write(grid_file, format=file_format)

    @property
    def grid_keys(self):
        return self.grid.colnames

    def unique_grid_values(self, key):
        return np.array(np.unique(self.grid[key]))

    def filter_grid(self, key, value):
        idx = np.where(self.grid[key] == value)
        return self.grid[idx]

    def _dr_old(self, x0, y0, z0):
        wx, wy, wz = self.weights
        dx = wx * abs(self.x - x0) / x0
        dy = wy * abs(self.y - y0) / y0
        dz = wz * abs(self.z - z0) / z0
        dr = dx + dy + dz
        return dr

    def _dr(self, x0, y0, z0):
        wx, wy, wz = self.weights

        # Standardize the data
        std_x = np.std(self.x)
        std_y = np.std(self.y)
        std_z = np.std(self.z)

        dx = wx * (self.x - x0) / std_x
        dy = wy * (self.y - y0) / std_y
        dz = wz * (self.z - z0) / std_z

        dr = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)

        return dr

    def _validate_input(self, x0, y0, z0):
        assert np.min(self.x) <= x0 <= np.max(self.x), 'r_petro(eta=0.2) is outside of the range of the grid'
        assert np.min(self.y) <= y0 <= np.max(self.y), 'r_50 is outside of the range of the grid'
        assert np.min(self.z) <= z0 <= np.max(self.z), 'C2080 is outside of the range of the grid'

    def _closest_row(self, x0, y0, z0):
        if self.enforce_range:
            self._validate_input(x0, y0, z0)
        dr = self._dr(x0, y0, z0)
        return self.grid[dr.argmin()]

    def _get_corrected_row(self, p):
        x0, y0, z0 = self._get_xyz_from_p(p)
        return self._closest_row(x0, y0, z0)

    def correct(self, p):
        corrected_p = copy(p)
        corrected_p.epsilon = self.estimate_epsilon(p)
        return corrected_p

    def estimate_n(self, p):
        """
        Given the half light radius and c2080 computed using the default epsilon value,
        return an estimated sersic index n.
        """
        row = self._get_corrected_row(p)
        return row['n']

    def estimate_epsilon(self, p):
        """
        Given the half light radius and c2080 computed using the default epsilon value,
        return a corrected epsilon value.
        """
        row = self._get_corrected_row(p)
        epsilon_fraction = p.epsilon_fraction
        eta = p.eta

        if epsilon_fraction == 0.5:
            r_ep = row['c_r_eff']
        elif epsilon_fraction == 0.8:
            r_ep = row['c_r_80']
        elif epsilon_fraction == .99:
            r_ep = row['c_r_99']
        else:
            raise ValueError('Input epsilon_fraction={} is not supported, choose from [0.5, 0.8, 0.99]')

        if eta not in self.eta_keys.keys():
            raise ValueError('Input eta={} is not supported, choose from eta={}'.format(
                eta, list(self.eta_keys.keys())))

        r_p = row[self.eta_keys[eta]]
        epsilon = r_ep / r_p
        return epsilon

    def _plot_grid(self, x0=None, y0=None, z0=None, cmap='hot', target_c='blue',
                   cmap_key='n', colorbar_label=None, suptitle=None, axs=None):
        if axs is None:
            fig, axs = plt.subplots(1, 3, figsize=[5 * 3, 5])
        else:
            assert len(axs) == 3, "axs should be a list of 3 axis"
            fig = axs[0].figure

        cm = plt.cm.get_cmap(cmap)

        sim_n_list = self.grid[cmap_key]

        ax = axs[0]
        sc = ax.scatter(self.x, self.y, c=sim_n_list, vmin=0, vmax=max(sim_n_list) + 1, s=35, cmap=cm)
        ax.set_xlabel('$r_{{p}}(\eta=0.2)$')
        ax.set_ylabel('$r_{{50}}$')

        ax = axs[1]
        sc = ax.scatter(self.x, self.z, c=sim_n_list, vmin=0, vmax=max(sim_n_list) + 1, s=35, cmap=cm)
        ax.set_xlabel('$r_{{p}}(\eta=0.2)$')
        ax.set_ylabel('$C_{2080}$')

        ax = axs[2]
        sc = ax.scatter(self.y, self.z, c=sim_n_list, vmin=0, vmax=max(sim_n_list) + 1, s=35, cmap=cm)
        ax.set_xlabel('$r_{{50}}$')
        ax.set_ylabel('$C_{2080}$')

        if None not in [x0, y0, z0]:
            idx = self._dr(x0, y0, z0).argmin()
            cx, cy, cz = self.x[idx], self.y[idx], self.z[idx]

            axs[0].scatter(x0, y0, marker='o', s=200, ec=target_c, fc='None', lw=5)
            axs[1].scatter(x0, z0, marker='o', s=200, ec=target_c, fc='None', lw=5)
            axs[2].scatter(y0, z0, marker='o', s=200, ec=target_c, fc='None', lw=5)

            axs[0].plot([x0, cx], [y0, cy], marker='o', lw=5)
            axs[1].plot([x0, cx], [z0, cz], marker='o', lw=5)
            axs[2].plot([y0, cy], [z0, cz], marker='o', lw=5)

        fig.colorbar(sc, ax=axs, location='bottom', aspect=50, label=colorbar_label if colorbar_label else cmap_key)
        fig.suptitle(suptitle if suptitle else 'Petrosian Correction Grid')

        return fig, axs

    def plot_correction(self, p, cmap='hot', target_c='blue', cmap_key='n',
                        colorbar_label=None, suptitle=None, axs=None):
        x0, y0, z0 = self._get_xyz_from_p(p)
        fig, axs = self._plot_grid(x0=x0, y0=y0, z0=z0, cmap=cmap, cmap_key=cmap_key,
                                   colorbar_label=colorbar_label, target_c=target_c,
                                   suptitle=suptitle, axs=axs)
        return fig, axs
