# Release Notes

## Version 0.3.1
*March 8th 2022*

**PetroFit Improvements**

- Removed kcorrect from Docker file. [:pr:`77`, :user:`robelgeda`]
- Added badges to `README.rst`.
- Implemented `get_amplitude_at_r` function that calculates the amplitude of an image at an isophotal radius. [:pr:`81`, :user:`robelgeda`] 
- Removed outdated `fit_plane` function. [:pr:`84`, :user:`robelgeda`]
- Moved requirements from `environment.yml` to `requirements.txt`. Though both can be used to install PetroFit, `environment.yml` is recommended at this time. [:pr:`72`, :user:`robelgeda`]
- Upgraded circleci `resource_class` to `medium+`. [:pr:`62`, :user:`robelgeda`]
- Removed outdated `petrofit.utils.cutout` function. [:pr:`93`, :user:`robelgeda`]

**General bug fixes and small changes**

- Cleaned up the docs, introduced a `CHANGES.md`file. [:pr:`77`, :user:`robelgeda`]
- Add release procedure for developers. [:pr:`70`, :user:`robelgeda`]