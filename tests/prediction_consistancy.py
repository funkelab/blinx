import numpy as np
import jax
import jax.numpy as jnp
from jax import lax
import time
from promap.trace_model import TraceModel
from promap.fluorescence_model import EmissionParams, FluorescenceModel
import unittest

def p_x_i_given_z_i(x_i, z, mu_i, sigma_i2, mu_b, sigma_b2):

    x = _bring_in(x_i)

    mean_i = np.log(z * mu_i * np.exp(sigma_i2 / 2))

    mean_b = np.log(mu_b)

    mean = np.log(np.exp(mean_i) + np.exp(mean_b))
    sigma2 = sigma_i2 + sigma_b2

    result = _integrate_from_cdf(x, mean, sigma2)
        
    return result

def _bring_in(x):

    return jnp.log(x)

def _normal_cdf(x, mu, sigma2):
    # CDF of the normal function
        
    return 0.5 * (1 + jax.lax.erf((x - mu)/jnp.sqrt(2 * sigma2)))
    
def _integrate_from_cdf(x, mu, sigma):
    #Aproximates the integral of the normal distribution from x : x + 1/256
        
    a = _normal_cdf(x, mu, sigma**2)
    b = _normal_cdf(x + (1/256), mu, sigma**2)
    prob = jnp.abs(a - b)
        
    return prob

def p_trace_given_z_i(trace, z, mu_i, sigma_i2, mu_b, sigma_b2):
    x = jnp.log(trace)
    
    mean_i = jnp.log(z * mu_i * jnp.exp(sigma_i2 / 2))
    mean_b = jnp.log(mu_b)
    mean = jnp.log(np.exp(mean_i) + jnp.exp(mean_b))
    sigma2 = sigma_i2 + sigma_b2
    
    result = _integrate_from_cdf(x, mean, sigma2)
    
    return result

def vmap_p_x_given_z(x, y, mu_i, sigma_i2, mu_b, sigma_b2):
    zs = jnp.arange(0,y+1)
    x = jnp.expand_dims(x,0)
    
    p_trace_given_z_i_bound = lambda x, z: p_trace_given_z_i(x, z, mu_i, 
                                                             sigma_i2, mu_b, 
                                                             sigma_b2)
    
    return jax.vmap(p_trace_given_z_i_bound, in_axes=(1,None))(x, zs)

def scan_f(p_accumulated, p_emission, p_transition):
    ''' 
    p_accu:
        - accumulated probability from t=0 to t-1
        - vector shape (1 x Y)
        
    p_emission:
        - probability of observing X given state z
        - precalculated and stored in a array shape (t x y)
        
    p_transtion:
        - probability of transitioning from state z(t-1) to state z(t)
        - precalculated and stored in an array shape (y x y)
    
    '''
    temp = p_emission * jnp.matmul(p_transition, p_accumulated)
    scale_factor = 1 / jnp.sum(temp)
    prob_time_t = temp * scale_factor
    
    return prob_time_t, scale_factor


def forward_alg_jax(probs, transition_m, p_init):
    
    initial_values = p_init[:] * probs[:,0]
    scale_factor_initial = 1 / jnp.sum(initial_values)
    
    p_transition = transition_m
    
    scan_f_2 = lambda p_accumulated, p_emission: scan_f(p_accumulated, 
                                                        p_emission, 
                                                        p_transition)

    final, result = lax.scan(scan_f_2, p_init, probs.T)
    
    return -1*(np.sum(np.log(result)))

def _forward_alg(x_trace, y, transition_m, p_init, mu_i, sigma_i2, mu_b, sigma_b2):

    forward = np.zeros((y+1, len(x_trace)))
    scale_factors = np.zeros((len(x_trace)))

    # Initialize
    for s in range(y+1):
        forward[s, 0] = p_init[s] * \
            p_x_i_given_z_i(x_trace[0], s, mu_i, sigma_i2, mu_b, sigma_b2)
    scale_factors[0] = 1 / np.sum(forward[:, 0])
    forward[:, 0] = forward[:, 0] * scale_factors[0]

    # Propagate
    for t in range(1, len(x_trace)): 
        for state in range(y+1):
            forward[state, t] = _alpha(forward, x_trace, transition_m,
                                                y, state, t, mu_i, sigma_i2, mu_b, sigma_b2)

        scale_factors[t] = 1 / np.sum(forward[:, t])
        forward[:, t] = forward[:, t] * scale_factors[t]

    # Total likelyhood
    log_fwrd_prob = -1*(np.sum(np.log(scale_factors)))

    return log_fwrd_prob

def _alpha(forward, x_trace, transition_m, y, s, t, mu_i, sigma_i2, mu_b, sigma_b2):
    # recursive element of the forward algorithm
    p = 0
    for i in range(y+1):
        p += forward[i, (t-1)] * transition_m[i, s] * \
            p_x_i_given_z_i(x_trace[t], s, mu_i, sigma_i2, mu_b, sigma_b2)
    return p


class TestTraceModel(unittest.TestCase):
    def test_p_x_i_given_z_i(self):
        trace = np.load('trace.npy')
        y = 3
        mu_i = 100
        mu_b = 140
        sigma_i = np.sqrt(0.5)
        sigma_b = np.sqrt(0.1)
        sigma_i2 = sigma_i**2
        sigma_b2 = sigma_b**2
        
        probs_basic = np.zeros((y+1, len(trace)))
        for i,z in enumerate(range(y+1)):
            for j, x_i in enumerate(trace):
               probs_basic[i,j] = p_x_i_given_z_i(x_i, z, mu_i, sigma_i2, mu_b, sigma_b2)
               
        
        
        probs_vmap = np.asarray(vmap_p_x_given_z(trace, y, mu_i, sigma_i2, mu_b, sigma_b2)).T
        
        e_params = EmissionParams(mu_i=mu_i, sigma_i=sigma_i,
                                  mu_b=mu_b, sigma_b=sigma_b)
        
        f_model = FluorescenceModel(e_params)
        probs_f_model = np.zeros((y+1, len(trace)))
        for i,z in enumerate(range(y+1)):
            for j, x_i in enumerate(trace):
               probs_f_model[i,j] = f_model.p_x_i_given_z_i(x_i, z)
               
        probs_f_model = f_model.vmap_p_x_given_z(trace, y+1)
               
        self.assertAlmostEqual(probs_basic.all(), probs_f_model.all(), places=5)
        self.assertAlmostEqual(probs_vmap.all(), probs_f_model.all(), places=5)
        
    def test_forward_alg(self):
        
        trace = np.load('trace.npy')
        y = 3
        mu_i = 100
        mu_b = 140
        sigma_i = np.sqrt(0.5)
        sigma_b = np.sqrt(0.1)
        sigma_i2 = sigma_i**2
        sigma_b2 = sigma_b**2
        e_params = EmissionParams(mu_i=mu_i, sigma_i=sigma_i,
                                  mu_b=mu_b, sigma_b=sigma_b)
        t_model = TraceModel(e_params, 0.1, len(trace))
        t_model.set_params(0.05, 0.1)
        
        
        probs_vmap = np.asarray(vmap_p_x_given_z(trace, y, mu_i, sigma_i2, mu_b, sigma_b2)).T
        
        prob_t_model = t_model.p_trace_given_y(trace, y)
        
        p_initial, transition_m = t_model._markov_trace(y)
        prob_basic = _forward_alg(trace, y, transition_m, p_initial, mu_i, sigma_i2, mu_b, sigma_b2)
        
        prob_jax = forward_alg_jax(probs_vmap, transition_m, p_initial)
        
        self.assertAlmostEqual(prob_basic, prob_t_model, places=0)
        self.assertAlmostEqual(prob_jax, prob_t_model, places=-1)  
        
        
        
if __name__ == "__main__":
    unittest.main()