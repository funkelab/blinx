blinx
=====

Installation
------------

With GPU support:

```bash
# create a conda environment with CUDA/CuDNN and JAX
conda create -n blinx python cudatoolkit=11.4 cudatoolkit-dev=11.4 cudnn=8.2 -c conda-forge
conda activate blinx
pip install 'jax==0.4.1' 'jaxlib==0.4.1+cuda11.cudnn82' -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# install ProMap
git clone https://github.com/funkelab/blinx
cd blinx
pip install .
```
