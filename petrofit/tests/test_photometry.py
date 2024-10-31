import numpy as np

from astropy.io import fits

import petrofit as pf

from matplotlib import pyplot as plt


def test_plot_apertures():
    """Simple test to plot apertures"""
    image = np.ones((50, 50))
    a1 = pf.radial_elliptical_aperture(position=(25, 25), r=10, elong=2, theta=45)
    a2 = pf.radial_elliptical_aperture(position=(25, 25), r=100, elong=10, theta=45)
    apertures = [a1, a2]

    pf.plot_apertures(image, apertures=apertures, vmin=0, vmax=2, color="red", lw=1.5)
    plt.show()

    # No aperture
    pf.plot_apertures(image, apertures=[], vmin=0, vmax=2, color="red", lw=1.5)
    plt.show()

    # No image
    pf.plot_apertures(None, apertures=apertures, vmin=0, vmax=2, color="red", lw=1.5)
    plt.show()

    # No aperture and no image
    pf.plot_apertures(None, apertures=[], vmin=0, vmax=2, color="red", lw=1.5)
    plt.show()


def test_radial_elliptical_aperture():
    """Test loading elliptical aperture"""
    image = np.ones((50, 50))
    a = pf.radial_elliptical_aperture(position=(25, 25), r=10, elong=2, theta=45)
    area = a.do_photometry(image)[0]
    ellip_area = np.pi * 10 * 5
    assert abs(area - ellip_area) / ellip_area < 0.01
    del image, a


def test_radial_elliptical_annulus():
    """Test loading elliptical annulus"""
    image = np.ones((50, 50))
    a = pf.radial_elliptical_annulus(position=(25, 25), r=10, dr=10, elong=2, theta=45)
    area = a.do_photometry(image)[0]
    ellip_area = (np.pi * 20 * 10) - (np.pi * 10 * 5)
    assert abs(area - ellip_area) / ellip_area < 0.01
    del image, a


def test_radial_photometry():
    """
    Test photometry core by doing photometry
    where the total flux = pixel area.
    """
    image_size = 30
    ones_image = np.ones((30, 30))

    r_list = pf.make_radius_list(image_size // 2, image_size // 2)

    results = pf.radial_photometry(
        ones_image, np.array(ones_image.shape) // 2, r_list, plot=True
    )

    plt.show()

    results = [r_list] + list(results)

    results = [np.round(l, 6) for l in results]

    for r, photometry, aperture_area, error in zip(*results):
        area = np.round(np.pi * r**2, 6)

        photo_error = abs(photometry - area) / area
        assert photo_error < 0.01, "{}".format(photo_error)

        area_error = abs(aperture_area - area) / area
        assert area_error < 0.01, "{}".format(area_error)
