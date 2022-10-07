import jax.numpy as jnp
import jax
from promap.trace_model import TraceModel
from promap.fluorescence_model import EmissionParams
import optax


class fit_trace:
    
    def __init__(self):
        self.trace_model = 
        return
    
    
    def liklihood_func(self, y):
        
        def unit(p_on, p_off, mu, sigma, trace):
            e_params = EmissionParams(mu_i = mu, sigma_i = sigma, mu_b=200, sigma_b=0.05)
            t_model = TraceModel(e_params, 0.1, 4000)
        
            probs = t_model.fluorescence_model.vmap_p_x_given_z_lognorm(trace, y)
            
            comb_matrix = t_model._create_comb_matrix(y)
            comb_matrix_slanted = t_model._create_comb_matrix(y, slanted=True)
            
            c_transition_matrix_2 = lambda p_on, p_off : t_model.create_transition_matrix(y, p_on, p_off,
                                                                       comb_matrix,
                                                                       comb_matrix_slanted)
            transition_matrix = c_transition_matrix_2(p_on, p_off)
            p_initial = transition_matrix[0,:]
            likelyhood = t_model._forward_alg_jax(probs, transition_matrix, p_initial)
            return  -1 * likelyhood  # need to flip to positive value for grad descent
        
        unit_grad_jit = jax.jit(jax.value_and_grad(unit, argnums=(0,1,2,3)))
        
        return unit_grad_jit
    
    def optimize_params(self, y, trace,
                        p_on_guess = 0.1,
                        p_off_guess = 0.1,
                        mu_guess = 50.,
                        sigma_guess = 0.2):
        
        # TODO: stop it from letting parameters go negative
        # TODO: add multiple starting points 
        # TODO: change learning rate for different parameters
        
        mu_grad_jit = self.liklihood_func(y)
        
        params = (p_on_guess, p_off_guess, mu_guess, sigma_guess)
        optimizer = optax.adam(learning_rate=1e-4)
        opt_state = optimizer.init(params)
        
        old_likelyhood = 1
        diff = 10
        p_on = p_on_guess
        p_off = p_off_guess
        mu = mu_guess
        sigma = sigma_guess
        
        while diff > 1e-4:
            
            likelyhood, grads = mu_grad_jit(p_on, p_off, mu, sigma, trace)
            #print(grads)
            
            
            updates, opt_state = optimizer.update(grads, opt_state)
            
            p_on, p_off, mu, sigma = optax.apply_updates((p_on, p_off, mu, sigma), 
                                                         updates)
            #print(f'{mu}, {sigma}, {p_on}, {p_off}')
            diff = jnp.abs(likelyhood - old_likelyhood)
            old_likelyhood = likelyhood
            #print(likelyhood)
            
        return -1*likelyhood, p_on, p_off, mu, sigma
    
    
    
    
    
    