# Release Notes


## Version 0.4.0
*Dev*

**PetroFit Improvements**

- Removed outdated `petrofit.utils.cutout` function. [:pr:#93]
- `model_to_image` function upgraded to use `astropy.convolution.utils.discretize_model`. [:pr:#95]
- `PSFModel` has been upgraded to `PSFConvolvedImageModel`.`PSFConvolvedImageModel` uses `__init__` function to wrap models. `PSFModel` class has been deprecated and will be removed in `v0.5`. [:pr:#100]
- An `origin` parameter has been added to `petrofit.models.make_grid`. [:pr:#100]

## Version 0.3.1
*March 8th 2022*

**PetroFit Improvements**

- Removed kcorrect from Docker file. [:pr:#77]
- Added badges to `README.rst`.
- Implemented `get_amplitude_at_r` function that calculates the amplitude of an image at an isophotal radius. [:pr:#81] 
- Removed outdated `fit_plane` function. [:pr:#84]
- Moved requirements from `environment.yml` to `requirements.txt`. Though both can be used to install PetroFit, `environment.yml` is recommended at this time. [:pr:#72]
- Upgraded circleci `resource_class` to `medium+`. [:pr:#62]

**General bug fixes and small changes**

- Cleaned up the docs, introduced a `CHANGES.md`file. [:pr:#77]
- Add release procedure for developers. [:pr:#70]