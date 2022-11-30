import jax.numpy as jnp
import jax
from promap.trace_model import TraceModel
from promap.fluorescence_model import EmissionParams
from promap import transition_matrix
import optax

def optimize_params(y, trace,
                    p_on_guess=0.1,
                    p_off_guess=0.1,
                    mu_guess=50.,
                    sigma_guess=0.2,
                    mu_b_guess=200):

    '''
    Use gradient descent to fit kinetic (p_on / off) and emission
    (mu / sigma) parameters to an intensity trace, for a given value of y

    Args:
        y (int):
            - The maximum number of elements that can be on.

        trace (jnp array):
            - ordered array of intensity observations
            - shape (number_observations, )

        p_on_guess / p_off_guess (float):
            - value between 0 and 1 not inclusive

        mu_guess (float):
            - the mean intensity on a single fluorophore when on

        sigma_guess (float):
            - the variance of the intensity of a single fluorophore

    Returns:
        The maximum log-likelihood that the trace arrose from y elements,
        as well as the optimum values of p_on, p_off, mu, and sigma
    '''

    # TODO: stop it from letting parameters go negative
    # TODO: add multiple starting points
    # TODO: change learning rate for different parameters
    #       - currently way to low for mu
    #       - add a seperate optimizer?
    # when underestimating count, training loop reaches a point where only mu is changing

    likelihood_grad_func = _create_likelihood_grad_func(y, mu_b_guess)
                                           # creates a new loss function
                                           # for the given y value

    params = (p_on_guess, p_off_guess, mu_guess, sigma_guess)
    optimizer = optax.adam(learning_rate=1e-3, mu_dtype='uint64')
    opt_state = optimizer.init(params)
    optax.keep_params_nonnegative() # works almost too well
    
    

    old_likelihood = 1
    diff = 10
    p_on = p_on_guess
    p_off = p_off_guess
    mu = mu_guess
    sigma = sigma_guess

    while diff > 1e-3:

        likelihood, grads = likelihood_grad_func(p_on, p_off, mu, sigma,
                                                 trace, mu_b_guess)
        print(f'{grads[0]:.2f}, {grads[1]:.2f}, {grads[2]:.2f}, {grads[3]:.2f}')
        
        updates, opt_state = optimizer.update(grads, opt_state)
        #print(updates)
        p_on, p_off, mu, sigma = optax.apply_updates((p_on, p_off, mu,
                                                      sigma), updates)

        diff = jnp.abs(likelihood - old_likelihood)
        old_likelihood = likelihood
        

        print(f'{likelihood:.2f}, {p_on:.4f}, {p_off:.4f}, {mu:.4f}, {sigma:.4f}')
        print('-'*50)
    return -1*likelihood, p_on, p_off, mu, sigma

def optimize_params_new(y, trace,
                    p_on_guess=0.1,
                    p_off_guess=0.1,
                    mu_guess=50.,
                    sigma_guess=0.2,
                    mu_b_guess=200,
                    mu_lr = 5):
    '''
    Use gradient descent to fit kinetic (p_on / off) and emission
    (mu / sigma) parameters to an intensity trace, for a given value of y

    Args:
        y (int):
            - The maximum number of elements that can be on.

        trace (jnp array):
            - ordered array of intensity observations
            - shape (number_observations, )

        p_on_guess / p_off_guess (float):
            - value between 0 and 1 not inclusive

        mu_guess (float):
            - the mean intensity on a single fluorophore when on

        sigma_guess (float):
            - the variance of the intensity of a single fluorophore

    Returns:
        The maximum log-likelihood that the trace arrose from y elements,
        as well as the optimum values of p_on, p_off, mu, and sigma
    '''
    
    likelihood_grad_func = _create_likelihood_grad_func(y, mu_b_guess)
                                           # creates a new loss function
                                           # for the given y value

    params = (p_on_guess, p_off_guess, mu_guess, sigma_guess)
    optimizer = optax.adam(learning_rate=1e-3, mu_dtype='uint64')
    opt_state = optimizer.init(params)
    
    mu_optimizer = optax.sgd(learning_rate=mu_lr)
    mu_opt_state = mu_optimizer.init(params[2])
    
    old_likelihood = 1
    diff = 10
    p_on = p_on_guess
    p_off = p_off_guess
    mu = mu_guess
    sigma = sigma_guess
    #update_scale = (jnp.asarray([1, 1, 1000, 1]))
    #mus = jnp.asarray([mu_guess])

    while diff > 1e-4:

        likelihood, grads = likelihood_grad_func(p_on, p_off, mu, sigma,
                                                 trace, mu_b_guess)
        
        updates, opt_state = optimizer.update(grads, opt_state)
        new_update = (updates[0], updates[1], updates[2], updates[3])
        
        mu_update, mu_opt_state = mu_optimizer.update(grads[2], mu_opt_state)
        
        p_on, p_off, mu, sigma = optax.apply_updates((p_on, p_off, mu,
                                                      sigma), new_update)
        mu = optax.apply_updates((mu), mu_update)
    
        diff = jnp.abs(likelihood - old_likelihood)
        old_likelihood = likelihood

        print(f'{likelihood:.2f}, {p_on:.4f}, {p_off:.4f}, {mu:.4f}, {sigma:.4f}')
        print('-'*50)
        
    return -1*likelihood, p_on, p_off, mu, sigma

def _create_likelihood_grad_func(y, mu_b_guess=200):
    '''
    Helper function that creates a loss function used to fit parameters
    p_on, p_off, mu, and simga

    Args:
        y (int):
            - The maximum number of elements that can be on

    Returns:
        a jited function that returns the likelihood, which acts as a loss,
        when given values for p_on, p_off, mu, and sigma
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
