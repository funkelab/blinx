.. blinx documentation master file

What is ``blinx`` ?
====================

``blinx`` is a tool designed to estimate the number of independantly blinking 
fluorescent light emitters,
when only their combined intensity contribuions can be observed.

Observing this blinking behaviour over time produces a 'trace'

.. jupyter-execute::
  :hide-code:

  import blinx
  from blinx.parameters import Parameters
  from blinx.trace_model import generate_trace

  parameters = Parameters(mu=2, mu_bg=1, sigma=0.001, p_on=0.1, p_off=0.1)
  trace, zs = generate_trace(y=5, parameters=parameters, num_frames=100)



Full Documentation:
===================

.. toctree::
  :maxdepth: 2

  install
  quickstart
  extending
  learning
  api
