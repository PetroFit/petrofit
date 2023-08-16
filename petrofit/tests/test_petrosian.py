import os
import pytest
from copy import deepcopy


import numpy as np

from astropy.table import Table
from matplotlib import pyplot as plt

import petrofit as pf


@pytest.fixture
def photutils_cog(data_dir):
    pg = Table.read(os.path.join(data_dir, 'photutils_cog.fits.gz'))
    areas = pg['A']
    fluxes = pg['L']
    areas_err = abs(np.random.normal(loc=0.0, scale=np.sqrt(areas.min()), size=areas.shape))
    fluxes_err = abs(np.random.normal(loc=0.0, scale=np.sqrt(fluxes.min()), size=areas.shape))
    pg.add_column(areas_err, name='A_err')
    pg.add_column(fluxes_err, name='L_err')
    return pg


def test_calculate_petrosian(photutils_cog):
    # Sample data
    areas = photutils_cog['A']
    fluxes = photutils_cog['L']
    areas_err = photutils_cog['A_err']
    fluxes_err = photutils_cog['L_err']

    # Without errors
    petrosian, err = pf.calculate_petrosian(areas, fluxes)
    assert len(petrosian) == len(areas)
    assert err is None

    # With errors
    petrosian, err = pf.calculate_petrosian(areas, fluxes, areas_err, fluxes_err)
    assert len(petrosian) == len(areas)
    assert len(err) == len(areas)
    assert np.isnan(petrosian[0])
    assert np.isnan(err[0])
    assert not np.any(np.isnan(petrosian[1:]))
    assert not np.any(np.isnan(err[1:]))

    # With only one type of error
    petrosian, err = pf.calculate_petrosian(areas, fluxes, None, fluxes_err)
    assert len(petrosian) == len(areas)
    assert len(err) == len(areas)

    petrosian, err = pf.calculate_petrosian(areas, fluxes, areas_err, None)
    assert len(petrosian) == len(areas)
    assert len(err) == len(areas)

    # Insufficient data points
    try:
        pf.calculate_petrosian([1, 2], [1, 2])
        assert False, "Expected an AssertionError due to insufficient data points"
    except AssertionError:
        pass

    # Test unsorted areas
    try:
        pf.calculate_petrosian([1, 3, 2, 4], [1, 6, 3, 10])
        assert False, "Expected an AssertionError due to unsorted areas"
    except AssertionError:
        pass


def test_petrosian(photutils_cog):
    # Sample data
    r_list = photutils_cog['r']
    area_list = photutils_cog['A']
    flux_list = photutils_cog['L']
    areas_err = photutils_cog['A_err']
    fluxes_err = photutils_cog['L_err']

    p = pf.Petrosian(r_list, area_list, flux_list, area_err=areas_err, flux_err=fluxes_err,)

    # Assert default
    assert p.epsilon == 2.
    assert p.eta == 0.2
    assert p.fraction_flux_to_r(0.5) == p.r_half_light
    assert p.fraction_flux_to_r(0.2) == p.concentration_index()[0]
    assert p.fraction_flux_to_r(0.8) == p.concentration_index()[1]

    # Check values
    assert abs(p.r_petrosian - 65) / 65 < 0.05
    assert abs(p.r_total_flux - 130) / 130 < 0.05
    assert abs(p.r_half_light - 30) / 30 < 0.05
    assert abs(p.total_flux - 408.84) / 408.84 < 0.05
    assert abs(p.c2080 - 2.777) / 2.777 < 0.05

    # Test if plotting works
    p.plot(True, True)
    plt.show()

    p.imshow()
    plt.show()

    p.plot_cog()
    plt.show()


def test_corrections(data_dir, photutils_cog):
    # Sample data
    r_list = photutils_cog['r']
    area_list = photutils_cog['A']
    flux_list = photutils_cog['L']
    areas_err = photutils_cog['A_err']
    fluxes_err = photutils_cog['L_err']

    p = pf.Petrosian(r_list, area_list, flux_list, area_err=areas_err, flux_err=fluxes_err, )

    pc = pf.PetrosianCorrection.read(
            os.path.join(
                data_dir,
                'corr.csv'
            )
    )

    pc.enforce_range = False

    n = pc.estimate_n(p)

    n_err = abs(n - 1) / 1
    assert n_err < 0.05  # 5% error

    epsilon = pc.estimate_epsilon(p)

    assert abs(epsilon - 1.7) / 1.7 < 0.05

    corrected_p = pf.Petrosian(
        p.r_list, p.area_list, p.flux_list,
        epsilon=epsilon
    )

    corrected_p.plot(True, True)
    plt.show()

    assert corrected_p.r_petrosian == p.r_petrosian
    assert abs(corrected_p.r_half_light - 30) / 30 < 0.05


def test_grid_gen(data_dir, output_dir):
    out_path = os.path.join(
                output_dir,
                "test_grid.csv"
    )
    pg = pf.generate_petrosian_sersic_correction(out_path,
                                                 r_eff_list=[10, 20, 30],
                                                 n_list=[0.5, 1, 4],
                                                 n_cpu=None,
                                                 psf=None,
                                                 oversample=('x_0', 'y_0', 50, 100),
                                                 ipython_widget=True,
                                                 overwrite=True, plot=False)

    pc = pf.PetrosianCorrection.read(
            os.path.join(
                data_dir,
                'corr.csv'
            )
    )

    for colname in pg.colnames:
        assert np.all(abs(pg[colname] - pc.grid[colname]) / pg[colname] < 0.01)






