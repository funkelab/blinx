import jax.numpy as jnp
from promap.trace_model import TraceModel
from promap.fluorescence_model import EmissionParams
from promap.fit_params import FitTrace
import matplotlib.pyplot as plt
import numpy as np
import time


if __name__ == '__main__':
    
    # generate a test trace
    y_test = 5
    seed = 100
    e_params = EmissionParams(mu_i = 50, sigma_i = 0.03, mu_b=200, sigma_b=0.15)
    t_model_t = TraceModel(e_params, 0.1, 4000)
    t_model_t.set_params(0.05, 0.05)
    x_trace, states = t_model_t.generate_trace(y_test, seed=seed)
    
    plt.plot(x_trace[:1000])
    plt.show()
    
    fit_functions = FitTrace()
    
    max_likelyhood = -1e8
    best_y = None
    
    # Calc likelyhood that trace arrose from different y values
    ys = np.arange(3,11)
    start = time.time()
    for y in ys:
        likelyhood, p_on, p_off, mu, sigma = fit_functions.optimize_params(y, x_trace)
        
        print('- '*20)
        print(f'y = {y}')
        print(f'log likelyhood   = { likelyhood:.2f}')
        print(f'p_on / p_off     = { p_on:.4f} / {p_off:.4f}')
        print(f'mu / sigma       = {mu:.4f} / {sigma:.4f}')   
        
        if likelyhood > max_likelyhood:
            max_likelyhood = likelyhood
            best_y = y
    
    print('* '*20)
    print(f'tested {len(ys)} y values in {time.time()-start:.2f}s')
    print(f'maximum likelyhood y = {best_y}')
        
    
    