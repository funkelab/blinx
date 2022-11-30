from promap.trace_model import TraceModel
from promap.fluorescence_model import EmissionParams
from promap import fit
import matplotlib.pyplot as plt
import numpy as np
import time
from jax.config import config
config.update("jax_enable_x64", True)


if __name__ == '__main__':

    # generate a test trace
    y_test = 50
    seed = 100
    e_params = EmissionParams(mu_i=50, sigma_i=0.03, mu_b=200, sigma_b=0.15)
    t_model_t = TraceModel(e_params, p_on=0.05, p_off=0.05)
    x_trace, states = t_model_t.generate_trace(y_test, seed=seed,
                                               num_frames=4000)

    plt.plot(x_trace[:1000])
    plt.show()


    
    y = 50
    start = time.time()
    likelihood, p_on, p_off, mu, sigma = fit.optimize_params(y, x_trace)

    print('- '*20)
    print(f'log likelihood   = { likelihood:.2f}')
    print(f'p_on / p_off     = { p_on:.4f} / {p_off:.4f}')
    print(f'mu / sigma       = {mu:.4f} / {sigma:.4f}')


