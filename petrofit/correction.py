from copy import copy
import yaml

import numpy as np

from matplotlib import pyplot as plt

from scipy.interpolate import interp1d
from scipy.special import gammaincinv

from astropy.utils.console import ProgressBar
from astropy.modeling import models

from .fitting.models import PSFModel, sersic_enclosed, sersic_enclosed_inv
from .photometry import photometry_step
from .fitting.fitting import model_to_image
from .petrosian import Petrosian


def generate_petrosian_sersic_correction(psf=None, r_eff_list=None, n_list=None, oversample=('x_0', 'y_0', 10, 10),
                                         plot=True, output_yaml_name=None):

    if r_eff_list is None:
        r_eff_list = np.arange(10, 100, 5)

    if n_list is None:
        n_list = np.arange(1, 6., .5)

    all_n_data = {}

    print("Start...")
    with ProgressBar(len(r_eff_list) * len(n_list)) as bar:
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

                r_total_flux = sersic_enclosed_inv(
                    total_flux,
                    amplitude=amplitude,
                    r_eff=r_eff,
                    n=n
                )
                r_total_flux = r_total_flux * 1.5
                print(r_eff, n, r_total_flux)

                image_size = int(r_total_flux * 2)
                x_0 = image_size // 2
                y_0 = image_size // 2

                # Define model
                galaxy_model = models.Sersic2D(
                    amplitude=amplitude,
                    r_eff=r_eff,
                    n=n,
                    x_0=x_0 + 0.5,
                    y_0=y_0 + 0.5,
                    ellip=0.,
                    theta=0.,
                )

                # PSF weap
                galaxy_model = PSFModel.wrap(galaxy_model, psf=psf, oversample=oversample)

                galaxy_image = model_to_image(x_0, y_0, image_size, galaxy_model)

                # Photometry on model image
                if r_total_flux >= 200:
                    r_list = [x for x in range(1, 201, 2)]
                    r_list += [x for x in range(250, int(r_total_flux) + 50, 50)]
                else:
                    r_list = [x for x in range(1, int(r_total_flux) + 2, 2)]

                flux_list, area_list, err = photometry_step((x_0, y_0), r_list, galaxy_image, plot=plot,
                                                            vmax=amplitude / 100)
                plt.show()

                # Petrosian from Photometry
                p = Petrosian(r_list, area_list, flux_list)

                rc1, rc2, c_index = p.concentration_index()
                if np.any(np.isnan(np.array([rc1, rc2, c_index]))):
                    raise Exception("concentration_index cant be computed")

                # Compute new r_total_flux
                f = interp1d(flux_list, r_list, kind='cubic')
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
                    print(r_total_flux, p.r_total_flux, corrected_p.r_total_flux, corrected_r_total_flux)
                    print(" ")

                c_pet_list_output.append(p.concentration_index()[-1])
                n_list_output.append(n)
                r_p_list_output.append(r_eff)
                epsilon_list_output.append(corrected_epsilon)
                r_p_hl_list_output.append(corrected_p.r_half_light)

            all_n_data[r_eff] = {
                'c_index': c_pet_list_output,
                'n': n_list_output,
                'r_petrosian': r_p_list_output,
                'epsilon': epsilon_list_output,
                'r_hl_petrosian': r_p_hl_list_output
            }

    if output_yaml_name is None:
        output_yaml_name = "sersic_correction_grid.yaml"

    with open(output_yaml_name, 'w') as outfile:
        print(outfile.name)
        yaml.dump(all_n_data, outfile)
