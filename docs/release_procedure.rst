#################
Release Procedure
#################

This section describes the release procedure for PetroFit.

Release Steps
*************

1. Navigate to the PetroFit `releases page <https://github.com/PetroFit/petrofit/releases>`_ and draft a new release.
This will automatically create a git tag that can be used to create the new release. For the title of the release use,
for example, `Version 0.3` or `Version 0.3.2` for minor releases. You should name the tag `v0.3` or `v0.3.2` for these
examples. All releases should come from the main branch.

2. To upload to PyPi, first clone or pull the `main` branch of the repository. Make sure that all the tags have been fetched
via:

.. code-block:: bash

    git fetch --tags

3. Create a source distribution `sdist` using `setup.py`:

.. code-block:: bash

    python setup.py sdist

4. Navigate to the `dist/` folder and identify the `sdist` file. **Make sure the version is correct**.

5. Upload to PyPi using (for example `v0.3.2`):

.. code-block:: bash

    twine upload petrofit-0.3.2.tar.gz
