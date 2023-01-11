ProMap
======

Installation
------------

With GPU support:

```bash
# create a conda environment with CUDA/CuDNN and JAX
conda create -n promap python cudatoolkit=11.4 cudatoolkit-dev=11.4 cudnn=8.2 -c conda-forge
conda activate promap
pip install 'jax==0.4.1' 'jaxlib==0.4.1+cuda11.cudnn82' -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# install ProMap
git clone https://github.com/funkelab/promap
cd promap
pip install -r requirements.txt
pip install .
```
