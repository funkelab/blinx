import jax.numpy as jnp


eps = jnp.finfo(jnp.float32).tiny


'''
Defines global constants for indexing in parameter and results vectors
'''
PARAM_MU = 0
PARAM_MU_BG = 1
PARAM_SIGMA = 2
PARAM_P_ON = 3
PARAM_P_OFF = 4

