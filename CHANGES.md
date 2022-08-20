# Release Notes

## Version 0.4.1
*TBD*

**PetroFit Enhancements**

- `CITATION.rst` added to provide citing and credit information. [#130]
- `segm_mask` can now take a list of labels which means it is possible to get multiple unmasked sources using `masked_segm_image` [#143]
- Limits of param bounds in `get_default_sersic_bounds` changed.  [#143]
    - `r_eff` min bound set to `1e-3`.
    - `n` min bound set to `0.1`.
    - `ellip` max bound set to `0.99`.
- Add `GenSersic2D` model, which is a Sersic model with generalized ellipse isophot (boxy) parameter. [#145]
- Implement `PetroApprox` which is a class with no-PSF polynomials which relate profile paramters such as C2080 and Sersic n.  [#146]

**General bug fixes and small changes**

- Bug Fix: Example RMS image in docs was not the same size as the example data image. [#114]
- Bug Fix: Model attribute bug fix in  `PSFConvolvedModel2D`.  [#145]
- Print out PSF sum if it is not equal to 1. [#145]
- Bug Fix: Change `make_catalog` `kernel_size` parameter to odd number. [#148]

## Version 0.4.0
*March 26th 2022*

Upgrades to fitting sub-module, see below for details.

**PetroFit Enhancements**

- Removed outdated `petrofit.utils.cutout` function. [#93]
- `model_to_image` function upgraded to use `astropy.convolution.utils.discretize_model`. [#95]
- `PSFModel` has been upgraded to `PSFConvolvedModel2D`.`PSFConvolvedModel2D` uses `__init__` function to wrap models. `PSFModel` class has been deprecated and will be removed in `v0.5`. [#100]
- An `origin` parameter has been added to `petrofit.models.make_grid`. [#100]
- `fitting.py` and `models.py` have been moved into a new `modeling` folder. This allows for all modeling functions and classes to be imported as `from petrofit.modeling import ...`. [#100] 
- `petrofit.modeling.fitting.fit_model` can now accept fitting weights, so users can pass fitting weights using rms or error images. [#100]
- `petrofit.modeling.fitting.plot_fit` now displays 3 panel image of fits (input image, model-image, and residual). [#100]

## Version 0.3.1
*March 8th 2022*

**PetroFit Enhancements**

- Removed kcorrect from Docker file. [#77]
- Added badges to `README.rst`.
- Implemented `get_amplitude_at_r` function that calculates the amplitude of an image at an isophotal radius. [#81] 
- Removed outdated `fit_plane` function. [#84]
- Moved requirements from `environment.yml` to `requirements.txt`. Though both can be used to install PetroFit, `environment.yml` is recommended at this time. [#72]
- Upgraded circleci `resource_class` to `medium+`. [#62]

**General bug fixes and small changes**

- Cleaned up the docs, introduced a `CHANGES.md`file. [#77]
- Add release procedure for developers. [#70]