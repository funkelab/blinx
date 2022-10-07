import jax.numpy as jnp
import jax
from promap.trace_model import TraceModel
from promap.fluorescence_model import EmissionParams
from promap.transition_matrix import TransitionMatrix
import optax

class FitTrace:
    '''

    '''
    def __init__(self):
        return

    def optimize_params(self, y, trace,
                        p_on_guess=0.1,
                        p_off_guess=0.1,
                        mu_guess=50.,
                        sigma_guess=0.2):

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
            The maximum log-likelyhood that the trace arrose from y elements,
            as well as the optimum values of p_on, p_off, mu, and sigma
        '''

        # TODO: stop it from letting parameters go negative
        # TODO: add multiple starting points
        # TODO: change learning rate for different parameters
        #       - currently way to low for mu
        #       - add a seperate optimizer?

        mu_grad_jit = self._liklihood_func(y)  # creates a new loss function
                                               # for the given y value

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

            updates, opt_state = optimizer.update(grads, opt_state)

            p_on, p_off, mu, sigma = optax.apply_updates((p_on, p_off, mu,
                                                          sigma), updates)

            diff = jnp.abs(likelyhood - old_likelyhood)
            old_likelyhood = likelyhood


        return -1*likelyhood, p_on, p_off, mu, sigma
    
    def _liklihood_func(self, y):
        '''
        Helper function that creates a loss function used to fit parameters
        p_on, p_off, mu, and simga

        Args:
            y (int):
                - The maximum number of elements that can be on

        Returns:
            a jited function that returns the likelyhood, which acts as a loss,
            when given values for p_on, p_off, mu, and sigma
        '''

        def unit(p_on, p_off, mu, sigma, trace):
            e_params = EmissionParams(mu_i=mu, sigma_i=sigma, mu_b=200,
                                      sigma_b=0.05)
            t_model = TraceModel(e_params, 0.1, 4000)
            transition_funcs = TransitionMatrix()

            probs = t_model.fluorescence_model.vmap_p_x_given_z_lognorm(trace,
                                                                        y)

            comb_matrix = transition_funcs._create_comb_matrix(y)
            comb_matrix_slanted = transition_funcs._create_comb_matrix(y, slanted=True)

            c_transition_matrix_2 = lambda p_on, p_off: transition_funcs.create_transition_matrix(y, p_on, p_off,
                                                                       comb_matrix,
                                                                       comb_matrix_slanted)
            transition_matrix = c_transition_matrix_2(p_on, p_off)
            p_initial = transition_matrix[0, :]
            likelyhood = t_model._forward_alg_jax(probs, transition_matrix,
                                                  p_initial)
            return -1 * likelyhood  # need to flip to positive value for grad descent

        unit_grad_jit = jax.jit(jax.value_and_grad(unit, argnums=(0, 1, 2, 3)))

        return unit_grad_jit
