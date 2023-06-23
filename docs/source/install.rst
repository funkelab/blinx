.. _sec_install:

Installation
============

.. automodule:: blinx
   :noindex:

For basic installation

.. code-block:: bash

  conda create -n blinx python
  conda activate blinx
  git clone https://github.com/funkelab/blinx.git
  cd blinx
  pip install .

It is possible to have some difficulties with the installation of JAX on macOS systems
try:

.. code-block:: bash

  conda create -n blinx python
  conda activate blinx
  git clone https://github.com/funkelab/blinx.git
  cd blinx
  pip install .
  conda config --add channels conda-forge
  conda config --set channel_priority strict
  conda install jaxlib

For installation compatible with a GPU

.. code-block:: bash

  conda create -n blinx python cudatoolkit=11.4 cudatoolkit-dev=11.4 cudnn=8.2 -c conda-forge
  conda activate blinx
  pip install 'jax==0.4.1' 'jaxlib==0.4.1+cuda11.cudnn82' -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
