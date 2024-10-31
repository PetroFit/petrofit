############
Installation
############

There are multiple ways to install the PetroFit package. The latest release of PetroFit can be installed from `PyPi <https://pypi.org/project/petrofit>`_ using the ``pip install petrofit`` command (see the `<pip_>`_ section below). To install the latest developer version of `PetroFit`, please clone the `PetroFit GitHub repository <https://github.com/PetroFit/petrofit>`_
and install the package (see the `<For Developers_>`_ section below).

pip
****
PetroFit can be installed using pip as follows:

.. code-block:: bash

    pip install petrofit

hatch
******
PetroFit is comaptible with ``hatch``. If you would like to create a jupyter lab instance with PetroFit installed, 
you can use the following command: 

 .. code-block:: bash
    
    cd <dir you want to launch jupyter lab in>
    hatch run jupyter:lab


Conda
*****

We have included an ``environment.yml`` file for creating a
`Conda <https://docs.conda.io/projects/conda/en/latest/user-guide/getting-started.html>`_ environment. You can create an
environment as follows:

**Step 1:** Clone the PetroFit repo and ``cd`` into the cloned repo.

.. code-block:: bash

    git clone https://github.com/PetroFit/petrofit.git

    cd petrofit


**Step 2:** Create the conda environment.

.. code-block:: bash

    conda env create -f environment.yml

**Step 3:** Install and activate the `petrofit` environment.

.. code-block:: bash

    conda env create -f environment.yml

    source activate petrofit


For Developers
**************

For developers, we recommend setting up a conda environment and then using the following to install the developer version:

.. code-block:: bash

    git clone https://github.com/PetroFit/petrofit.git

    cd petrofit

    pip install -e .


If you will be contributing to the software, we recommend forking the repository on GitHub first, cloning your forked repository,
and then installing the developer version.

Frozen versions of conda environment files are provided via the `petrofit_environments repository <https://github.com/PetroFit/petrofit_environments>`_.

