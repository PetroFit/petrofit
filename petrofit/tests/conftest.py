import os
from glob import glob
import pytest

from astropy.io import fits


@pytest.fixture
def data_dir():
    """fixture for test data dir"""
    return os.path.join(os.path.dirname(__file__), "data")


@pytest.fixture
def output_dir():
    """fixture for test output dir"""
    path = os.path.join(os.path.dirname(__file__), "test_output_dir")
    if not os.path.isdir(path):
        os.mkdir(path)
    return path


@pytest.fixture
def galfit_images(data_dir):
    """fixture for galfit test images used for testing"""
    path = os.path.join(data_dir, "galfit_sersic*fits.gz")
    fb = glob(path)

    galfit_image_dict = {}
    for f in fb:
        name = os.path.basename(f).replace("galfit_sersic_", "").replace(".fits.gz", "")
        image = fits.getdata(f)
        galfit_image_dict[name] = image

    return galfit_image_dict


@pytest.fixture
def galfit_psf_images(data_dir):
    """fixture for galfit test images used for testing"""
    path = os.path.join(data_dir, "galfit_n4_psf*.gz")
    fb = glob(path)

    galfit_psf_image_dict = {}
    for f in fb:
        name = int(
            os.path.basename(f).replace("galfit_n4_psf", "").replace(".fits.gz", "")
        )
        image = fits.getdata(f)
        galfit_psf_image_dict[name] = image

    return galfit_psf_image_dict


@pytest.fixture
def psf_image(data_dir):
    """fixture for PSF test images used for testing"""
    path = os.path.join(data_dir, "f105w_psf.fits.gz")
    PSF = fits.getdata(path)
    PSF /= PSF.sum()
    return PSF
