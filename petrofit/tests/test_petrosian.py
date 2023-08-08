import os
import pytest
from copy import deepcopy

import numpy as np

from matplotlib import pyplot as plt

from petrofit.photometry import source_photometry, make_radius_list, order_cat

from petrofit.petrosian import Petrosian, PetrosianCorrection


def test_petrosian(sersic_petrosian):
    p = sersic_petrosian

    # Assert default
    assert p.epsilon == 2.
    assert p.eta == 0.2
    assert p.fraction_flux_to_r(0.5) == p.r_half_light
    assert p.fraction_flux_to_r(0.2) == p.concentration_index()[0]
    assert p.fraction_flux_to_r(0.8) == p.concentration_index()[1]

    # Check values
    assert (p.r_petrosian - 58.274255) / 58.274255 < 0.05
    assert (p.r_total_flux - 116.54851) / 116.54851 < 0.05
    assert (p.r_half_light - 24.233447) / 24.233447 < 0.05
    assert (p.total_flux - 9789.636481) / 9789.63648 < 0.05
    assert (p.concentration_index()[0] - 9.641128) / 9.641128 < 0.05

    # Test if plotting works
    p.plot(True, True)
    plt.show()

    p.imshow()
    plt.show()


def test_corrections(sersic_petrosian):
    return 
    p = sersic_petrosian

    pc = PetrosianCorrection.read(
        os.path.join(
            os.path.dirname(__file__),
            'concentration_index_grid_no_psf.yaml'
        )
    )

    c2080 = p.concentration_index()[-1]

    n = pc.estimate_n(p.r_half_light, c2080)

    n_err = (n - 2) / 2
    assert n_err < 0.05 # 5% error

    epsilon = pc.estimate_epsilon(p.r_half_light, c2080)

    assert (epsilon - 3.4) / 3.4 < 0.05

    corrected_p = Petrosian(
        p.r_list, p.area_list, p.flux_list,
        epsilon=epsilon
    )

    corrected_p.plot(True, True)
    plt.show()

    assert corrected_p.r_petrosian == p.r_petrosian
    assert (corrected_p.r_half_light - 25) / 25 < 0.05

    r_total_flux = p.r_total_flux

    p.epsilon = 4

    assert p.r_total_flux == r_total_flux * 2







