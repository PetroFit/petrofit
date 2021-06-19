from copy import copy
import yaml

import numpy as np

from matplotlib import pyplot as plt

from scipy.interpolate import interp1d
from scipy.special import gammaincinv

from astropy.utils.console import ProgressBar
from astropy.modeling import models

from petrofit.models import PSFModel, sersic_enclosed, sersic_enclosed_inv
from petrofit.photometry import photometry_step
from petrofit.fitting import model_to_image
from petrofit.petrosian import Petrosian


__all__ = ['generate_petrosian_sersic_correction']


def generate_petrosian_sersic_correction(output_yaml_name, psf=None, r_eff_list=None, n_list=None,
                                         oversample=('x_0', 'y_0', 10, 10), plot=True):
    """
    This function generates corrections for Petrosian radii by simulating a galaxy image and measuring its properties.
    This is done to identify the correct `epsilon` value that, multiplied with `r_petrosian`, would give `r_total_flux`.
    To achieve this, an image created from a Sersic model and convolved with a PSF (if provided). The Petrosian radii
    and concentrations are computed using the default `epsilon` = 2. Since the real `r_total_flux` of the simulated galaxy
    is known, the correct `epsilon` can be determined by `epsilon = r_petrosian / r_total_flux`. The resulting grid will
    be used to map measured `r_petrosian` and `C2080` to the correct `epsilon` value. After the gird is computed, it
    will be saved to a yaml file which is readable by `petrofit.petrosian.PetrosianCorrection`.

    Parameters
    ----------
    output_yaml_name : str
        Name of output file, must be .yaml or .yml

    psf : numpy.array
        2D PSF image to pass to `petrofit.fitting.models.PSFModel`

    r_eff_list : list, (optional)
        List of `r_eff` (half light radii) in pixels to evaluate.

    n_list : list, (optional)
        List of Sersic indices to evaluate.

    oversample : int or tuple
        oversampling to pass to `petrofit.fitting.models.PSFModel`

    plot : bool
        Shows plot of photometry and Petrosian

    Returns
    -------
    petrosian_grid : dict
        Dictionary that is readable by `petrofit.petrosian.PetrosianCorrection`
        Also saves Yaml file that is readable by `petrofit.petrosian.PetrosianCorrection`.
    """

    if r_eff_list is None:
        r_eff_list = np.arange(10, 100 + 5, 5)

    if n_list is None:
        n_list = np.arange(0.5, 6.0 + 0.5, 0.5)

    petrosian_grid = {}

    with ProgressBar(len(r_eff_list) * len(n_list), ipython_widget=True) as bar:

        for r_eff_idx, r_eff in enumerate(r_eff_list):

            c_pet_list_output = []
            n_list_output = []
            r_p_list_output = []
            epsilon_list_output = []
            r_p_hl_list_output = []

            for n_idx, n in enumerate(n_list):
                bar.update()

                n = np.round(n, 2)
                amplitude = 100 / np.exp(gammaincinv(2. * n, 0.5))

                # Total flux
                total_flux = sersic_enclosed(
                    np.inf,
                    amplitude=amplitude,
                    r_eff=r_eff,
                    n=n
                )
                total_flux = total_flux * 0.99

                # Total flux radius
                r_total_flux = sersic_enclosed_inv(
                    total_flux,
                    amplitude=amplitude,
                    r_eff=r_eff,
                    n=n
                )
                ori_r_total_flux = r_total_flux
                max_r = r_total_flux * 2 if n < 2 else r_total_flux * 1.2

                # Make r_list
                if max_r >= 200:
                    r_list = [x for x in range(1, 201, 2)]
                    r_list += [x for x in range(300, int(max_r) + 100, 100)]
                else:
                    r_list = [x for x in range(1, int(max_r) + 2, 2)]
                r_list = np.array(r_list)

                image_size = max(r_list) * 2

                x_0 = image_size // 2
                y_0 = image_size // 2

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

                # PSF weap
                galaxy_model = PSFModel.wrap(galaxy_model, psf=psf, oversample=oversample)

                galaxy_image = model_to_image(x_0, y_0, image_size, galaxy_model)

                flux_list, area_list, err = photometry_step((image_size // 2, image_size // 2), r_list, galaxy_image,
                                                            plot=plot,
                                                            vmax=amplitude / 100)
                plt.show()

                if plot:
                    print(r_eff, n, r_total_flux)
                    plt.show()

                # Petrosian from Photometry
                p = Petrosian(r_list, area_list, flux_list)
                rc1, rc2, c_index = p.concentration_index()
                if np.any(np.isnan(np.array([rc1, rc2, c_index]))):
                    raise Exception("concentration_index cant be computed")

                # Compute new r_total_flux
                u, indices = np.unique(flux_list, return_index=True)
                indices = np.array(indices)
                f = interp1d(flux_list[indices], r_list[indices], kind='linear')
                corrected_r_total_flux = f(total_flux)

                # Compute corrections
                corrected_epsilon = corrected_r_total_flux / p.r_petrosian
                corrected_p = copy(p)
                corrected_p.epsilon = corrected_epsilon
                if plot:
                    corrected_p.plot(True, True)
                    plt.show()
                    print(corrected_epsilon)
                    print(r_eff, p.r_half_light, corrected_p.r_half_light)
                    print(ori_r_total_flux, p.r_total_flux, corrected_p.r_total_flux, corrected_r_total_flux)
                    print(" ")

                c_pet_list_output.append(p.concentration_index()[-1])
                n_list_output.append(n)
                r_p_list_output.append(p.r_petrosian)
                epsilon_list_output.append(corrected_epsilon)
                r_p_hl_list_output.append(corrected_p.r_half_light)

                del galaxy_model, galaxy_image

            petrosian_grid[r_eff] = {
                'c_index': c_pet_list_output,
                'n': n_list_output,
                'epsilon': epsilon_list_output,

                'r_petrosian': r_p_list_output,
                'r_hl_petrosian': r_p_hl_list_output
            }

    if output_yaml_name is not None:
        with open(output_yaml_name, 'w') as outfile:
            print(outfile.name)
            yaml.dump(petrosian_grid, outfile)

    return petrosian_grid