import jax.numpy as jnp
import jax
import numpy as np
from promap.trace_model import TraceModel
from promap import fluorescence_model
from promap import transition_matrix
from promap import fit


if __name__ == '__main__':
    
    # Test that the function is able to fit parameters given correct y
    f_model = fluorescence_model.FluorescenceModel(sigma_i= 0.03, mu_i=2000, mu_b=5000)
    t_model = TraceModel(f_model, p_on= 0.05, p_off= 0.05)
    sim_trace_1, sim_states_1 = t_model.generate_trace(4, 45, 4000)
    
    likelihood_1, p_on_1, p_off_1, mu_1, sigma_1 = fit.optimize_params(4, sim_trace_1,
         mu_b_guess=5000,
         mu_guess=1500.,
         mu_lr=1)
    
    print(f'p_on: {p_on_1:.2f}, p_off: {p_off_1:.2f}, mu: {mu_1:.0f}')
    
    
    # test finding of initial guesses
    print('ESTIMATING INITIAL GUESSES')
    initial_guesses = fit._initial_guesses(100, 0.1, 4, sim_trace_1,
         mu_b_guess=5000, sigma=0.1)
    print(f'p_on: {initial_guesses[0][0]:.2f} \
    p_off: {initial_guesses[1][0]:.2f} \
    mu: {initial_guesses[2][0]:.2f}')
    
    
    