.. _sec_quickstart:

Quickstart
==========

.. automodule:: blinx
   :noindex:

:jupyter-download-notebook:`Download this page as a Jupyter notebook<quickstart>`


Generating a Trace
------------------

Simulating an intensity **trace** is easy in blinx. First we create an instance of :class:`parameters` and specify
the five parameters that define the kinetic and intensity parameters of our trace:
  * **mu**: the mean intensity of a signle 'on' emitter
  * **mu_bg**: the mean background intensity
  * **sigma**: the standard deviation in the intensity of a single on event
  * **p_on**: the probability of an 'off' emitter turning on at the next timepoint
  * **p_off**: the probability of an 'on' emitter turning off at the next timepoint

.. jupyter-execute::

  import blinx
  from blinx.parameters import Parameters
  from blinx.trace_model import generate_trace
  from blinx.parameter_ranges import ParameterRanges
  from blinx.hyper_parameters import HyperParameters
  from blinx.estimate import estimate_y
  import plotly.express as px
  import numpy as np

  parameters = Parameters(mu=1000, mu_bg=1000, sigma=0.03, p_on=0.05, p_off=0.05)

Next we specify the known number of emitters, and the number of observations (frames). Then pass these along with our parameters to
`generate_trace`. This returns a tuple containing the simualted intensity trace as well as the number of emitters 'on' in each observation.

.. jupyter-execute::

  num_emitters = 2
  num_observations = 500
  trace, states = generate_trace(num_emitters, parameters, num_observations)

  # plot the resulting trace
  fig = px.line(x=np.arange(0,num_observations), y=trace[0],
  labels= {'x': 'time', 'y': 'Intensity'},
  height=400, width=700)
  fig.show()

Define Optimization Parameters
------------------------------

Next we need to create two classes that define the search and optimization parameters 

:class:`ParameterRanges` defines the search space and density for the initial grid search of the five fitted parameters.

.. jupyter-execute::

  parameter_ranges = ParameterRanges(
    mu_range=(600, 1500),
    mu_bg_range=(600, 1500),
    sigma_range=(0.03, 0.05),
    p_on_range=(1e-3, 0.1),
    p_off_range=(1e-3, 0.1),
    mu_step=5,
    mu_bg_step=3,
    sigma_step=3,
    p_on_step=5,
    p_off_step=5)

:class:`HyperParameters` defines the hyperparameters used in the gradient descent of the five fitted parameters. For most 
arguments the defaults will be fine, however it is important to set the maximum expected intensity value (`max_x`) and to check the gradietn step sizes `step sizes`. As a guidline, the step sizes should be set to 1e-4  times the expected parameter value 
(i.e. for a 'mu' guess of 1000 a step size of 1e-1 is a good starting point)

.. jupyter-execute::

  hyper_parameters = HyperParameters(
  # step_sizes input is  Parameters class, 
  #but this is the gradient step size,
  # NOT the parameter guess
  step_sizes=Parameters(mu=1e-1, mu_bg=1e-1, sigma=1e-8, p_on=1e-6, p_off=1e-6),
  max_x=5000)

Fitting
-------

Now we are ready to fit the parameters and estiamte a count for our **Trace**.

.. jupyter-execute::

  y, parameters, likelihoods = estimate_y(
    trace,
    max_y = num_emitters + 1, # be sure to increase for experimental inference
    parameter_ranges=parameter_ranges,
    hyper_parameters=hyper_parameters)

  print(f'estimated N = {y[0][0]} emitters')


