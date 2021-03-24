############
Installation
############

To install the latest dev version of `PetroFit`, please clone the `PetroFit repo <https://github.com/PetroFit/petrofit>`_
and install the package. There are two ways to install the requirements needed to run the PetroFit package, through
`Conda <https://docs.conda.io/projects/conda/en/latest/user-guide/getting-started.html>`_ and
`Docker <https://docs.docker.com/get-docker/>`_.

Conda
*****

We have included a ``environment.yml`` file for creating a
`Conda <https://docs.conda.io/projects/conda/en/latest/user-guide/getting-started.html>`_ environment. You can create an
environment as follows:

**Step 1:** Clone the PetroFit repo and ``cd`` into the cloned repo.

.. code-block:: bash

    git clone https://github.com/PetroFit/petrofit.git

    cd petrofit


**Step 2:** Create the conda environment.

.. code-block:: bash

    conda env create -f environment.yml

**Step 3:** Activate the `petrofit` environment.

.. code-block:: bash

    source activate petrofit

**Step 4:** Install PetroFit.

.. code-block:: bash

    python setup.py install

|

Docker
******

We have included a Docker file as well as a helper script to make installation of the package as easy as possible.
The docker file will create a Jupyter Notebook image which makes creating and running Notebooks possible.
The Docker file also installs kcorrection software that might be useful. To install using the helper script, please
follow the instructions below.


**Step 1:** Clone the PetroFit repo and ``cd`` into the cloned repo.

.. code-block:: bash

    git clone https://github.com/PetroFit/petrofit.git

    cd petrofit

**Step 2:** Use the helper script to create the new docker image.

.. code-block:: bash

    python docker.py build


.. Note::

    The cloned repo will be mounted when running using the helper script and any changes to the host code will
    become available in the docker image (notebook restart may be required to import changed version).

**Step 3:** Start the notebook using the helper script. You can mount a host directory or a list of directories
(separated by space) by adding their paths at the end of the command. The host directories will be mounted under
the `mount` directory in the docker image.

.. code-block:: bash

    python docker.py run [extra_mount_paths]

.. important::

    Please note that this type of mounting is calling a ``bind`` which means the changes you make to the mounted directory
    will also apply in the host directory.

**Step 4:** Copy and paste the notebook URL into your internet browser. The link to the notebook home should look like:

.. code-block:: bash

    http://127.0.0.1:8888/?token=d020c13d029013c20d0329e6913c5df076d0a4a14e63dc77

**Step 5:** You can close the server like any other Jupyter notebook server by hitting ``Ctrl + C``
(make sure to have saved your notebook before shutting down the server).