import matplotlib.pyplot as plt
from promap import extract
from promap import fit
import jax.numpy as jnp
import jax
from promap.trace_model import TraceModel
from promap.fluorescence_model import EmissionParams
from promap import transition_matrix
import optax


def _create_likelihood_grad_func(y, mu_b_guess=200):
    '''
    Helper function that creates a loss function used to fit parameters
    p_on, p_off, mu, and simga
    '''

    def likelihood_func(p_on, p_off, mu, sigma, trace, mu_b_guess=200):
        e_params = EmissionParams(mu_i=mu, sigma_i=sigma, mu_b=mu_b_guess,
                                  sigma_b=0.05)
        t_model = TraceModel(e_params)

        probs = t_model.fluorescence_model.vmap_p_x_given_z_lognorm(trace,
                                                                    y)

        comb_matrix = transition_matrix._create_comb_matrix(y)
        comb_matrix_slanted = transition_matrix._create_comb_matrix(y, slanted=True)

        c_transition_matrix_2 = lambda p_on, p_off: transition_matrix.create_transition_matrix(y, p_on, p_off,
                                                                   comb_matrix,
                                                                   comb_matrix_slanted)
        transition_mat = c_transition_matrix_2(p_on, p_off)
        p_initial = transition_mat[0, :]
        likelihood = t_model.get_likelihood(probs, transition_mat,
                                              p_initial)
        return -1 * likelihood  # need to flip to positive value for grad descent

    unit_grad_jit = jax.jit(jax.value_and_grad(likelihood_func, argnums=(0, 1, 2, 3)))

    return unit_grad_jit



def get_likelihood(probs, transition_m, p_init):
    ''' 
    returns the likelihood of a trace, given the 4 parameters
    '''
    initial_values = p_init[:] * probs[:, 0]
    scale_factor_initial = 1 / jnp.sum(initial_values)
    initial_values = initial_values * scale_factor_initial
    p_transition = transition_m

    scan_f_2 = lambda p_accumulate, p_emission: _scan_likelihood(
                                                             p_accumulate,
                                                             p_emission,
                                                             p_transition)

    final, result = jax.lax.scan(scan_f_2, initial_values, probs.T)

    return -1*(jnp.sum(jnp.log(result))), result

def _scan_likelihood(p_accumulate, p_emission, p_transition):
    # scanning unit of lax.scan for get_likelihood
    temp = p_emission * jnp.matmul(p_accumulate, p_transition)
    scale_factor = 1 / jnp.sum(temp)
    prob_time_t = temp * scale_factor

    return prob_time_t, scale_factor

def vmap_p_x_given_z_lognorm(x, y, mu_i, mu_b, sigma_i):
    # calc the prob that each intensity arose from z bound flourophores
    zs = jnp.arange(0, y+1)
    x = jnp.expand_dims(x, 0)

    func_1 = lambda x, z: _p_trace_given_z_lognorm(x, z, mu_i, mu_b, sigma_i)
    
    result = jax.vmap(func_1,
                      in_axes=(1, None))(x, zs)

    return result.T

def _p_trace_given_z_lognorm(x, z, mu_i, mu_b, sigma_i):

    mean = jnp.log(mu_i * z + mu_b)
    value_1 = jnp.log(x)
    value_2 = value_1 + 1/256

    prob_1 = jax.scipy.stats.norm.cdf(value_1, loc=mean,
                                      scale=sigma_i)
    prob_2 = jax.scipy.stats.norm.cdf(value_2, loc=mean,
                                      scale=sigma_i)

    prob = jnp.abs(prob_1 - prob_2)

    return prob


if __name__ == '__main__':
    
    image_file_path = '../../promap/scripts/examples/example_image.tif'
    pick_file_path = '../../promap/scripts/examples/locs_picked.hdf5'
    drift_file_path = '../../promap/scripts/examples/drift.txt'

    trace = extract.extract_trace(image_file_path,
                                  pick_file_path,
                                  drift_file_path,
                                  spot_num=17,
                                  pixels=2)

    plt.plot(trace[1000:2000])
    
    y = 2
    
    likelihood, p_on, p_off, mu, sigma = fit.optimize_params(y, trace,
                                                              mu_guess = 2000.,
                                                              mu_b_guess=5000.,
                                                              mu_lr=5)
    
    # re-create the error
    mu_b_guess = 5000
    p_on = 0.0272
    p_off = 0.0298
    mu = 1815.1550
    sigma = 0.0713
    
    comb_matrix = transition_matrix._create_comb_matrix(y)
    comb_matrix_slanted = transition_matrix._create_comb_matrix(y, slanted=True)

    c_transition_matrix_2 = lambda p_on, p_off: transition_matrix.create_transition_matrix(y, p_on, p_off,
                                                               comb_matrix,
                                                               comb_matrix_slanted)
    transition_mat = c_transition_matrix_2(p_on, p_off)
    p_initial = transition_mat[0, :]

    # find timepoint where the nans start
    e_params = EmissionParams(mu_i=mu, sigma_i=sigma, mu_b=mu_b_guess,
                              sigma_b=0.05)
    t_model = TraceModel(e_params)

    probs = vmap_p_x_given_z_lognorm(trace, y, mu_i=mu, mu_b=mu_b_guess,
                                     sigma_i=sigma)
    
    likelihood, result_nan = get_likelihood(probs, transition_mat,
                                          p_initial)
    
