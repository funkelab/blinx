# blinx

- **[Introduction](#introduction)**
- **[Installation](#installation)**
- **[Examples](#examples)**
- **[Citation](#citation)**


# Introduction
This repository contains code to estimate the number of fluorescent emitters
when only their combined intensity can be measured, as reported in 
***A Bayesian Method to Count 
the Number of Molecules within a Diffraction Limited Spot***
[pre-print](https://www.biorxiv.org/content/10.1101/2024.04.18.590066v2) now available on BioRxiv.
Experiments associated with the pre-print can be found in [blinx_experiments](https://github.com/funkelab/blinx_experiments)


<img src="imgs/overview.png" />

Detailed documentation can be found: [here](https://funkelab.github.io/blinx/)

# Installation
For a basic CPU installation:
```bash
conda create -n blinx python
conda activate blinx
git clone https://github.com/funkelab/blinx.git
cd blinx
pip install .
```

For a GPU installation specific versions of jax and jaxlib must be pinned: 
```bash
conda create -n blinx python cudatoolkit=11.4 cudatoolkit-dev=11.4 cudnn=8.2 -c conda-forge
conda activate blinx
git clone https://github.com/funkelab/blinx.git
cd blinx
pip install .
pip install 'jax==0.4.1' 'jaxlib==0.4.1+cuda11.cudnn82' -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

```
# Examples
`blinx` can be used for both inference and simulation. The `estimate` module contains functions to run inference on an 
intensity trace and determine the molecular count, while the `generate_trace` function can be used to simulate traces 
with known parameters.
### Inference:

`blinx.estimate`
```python
import blinx
from blinx.estimate import estimate_y

traces = ... # load traces

# specify the range of an initial parameter grid search
parameter_ranges = blinx.ParameterRanges()
# Specify hyper-parameters
hyper_params = blinx.HyperParameters()

count, map_parameters, likelihood, evidence = estimate_y(
	traces=traces,
	max_y=..., # maximum count to test
	parameter_ranges=parameter_ranges,
	hyper_parameters=hyper_params)
```


### Simulation:

`blinx.trace_model.generate_trace`
```python
import blinx
from blinx.trace_model import generate_trace

# Specify kinetic and intensity parameters
parameters = blinx.parameters.Parameters()
# Specify hyper-parameters
hyper_params = blinx.HyperParameters()

sim_trace, sim_zs = generate_trace(
	y=4, # the number of emitters
	parameters=parameters,
	num_frames=4000, # length of the simulated trace
	hyper_parameters=hyper_params)

```

# Citation

If you use blinx in your research, please cite the BioRxiv pre-print:

```bash
@article{hillsley_bayesian_2024,
	title = {A Bayesian Solution to Count the Number of Molecules within a Diffraction Limited Spot},
	author = {Hillsley, Alexander and Stein, Johannes and Tillberg, Paul W. and Stern, David L. and Funke, Jan},
	doi = {10.1101/2024.04.18.590066},
	publisher = {{bioRxiv}},
	date = {2024-04-22},
}
```