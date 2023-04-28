import time

import jax
import jax.numpy as jnp
from jax import random
from scipy.special import comb

from .fluorescence_model import (
    create_emission_distribution,
    discretize_trace,
    sample_x_given_z,
)
from .markov_chain import get_measurement_log_likelihood, get_steady_state


def get_trace_log_likelihood(trace, y, parameters, hyper_parameters):
    """
    TODO: add a docstring
    """

    mu = parameters.mu
    mu_bg = parameters.mu_bg
    sigma = parameters.sigma
    p_on = parameters.p_on
    p_off = parameters.p_off

    p_transition = create_transition_matrix(y, p_on, p_off)
    p_initial = get_steady_state(p_transition)
    p_emission = create_emission_distribution(y, mu, mu_bg, sigma, hyper_parameters)

    discrete_trace = discretize_trace(trace, hyper_parameters)

    return get_measurement_log_likelihood(
        discrete_trace, p_emission, p_initial, p_transition
    )


def generate_trace(y, parameters, num_frames, seed=None):
    """generate a synthetic intensity trace

    Args:
        y (int):
            - the total number of fluorescent emitters

        parameters (array):
            - the parameters of the fluoresent and trace model

        num_frames (int):
            - the number of observations to simulate

        seed (int, optional):
            - random seed for the jax psudo rendom number generator

    Returns:
        x_trace (array):
            - an ordered array of length num_frames containing intensity
                values for each frame

        states (array):
            - array the same shape as x_trace, showing the hiddens state
                "z" for each frame
    """

    if seed is None:
        seed = time.time_ns()

    mu = parameters.mu
    mu_bg = parameters.mu_bg
    sigma = parameters.sigma
    p_on = parameters.p_on
    p_off = parameters.p_off

    p_transition = create_transition_matrix(y, p_on, p_off)
    p_initial = get_steady_state(p_transition)

    # jax.random.categorical takes log probs
    log_p_initial = jnp.log(p_initial)

    # generate a list of states, use scan b/c state t depends on state t-1
    key = random.PRNGKey(seed)
    key, subkey = random.split(key)
    initial_z = jnp.expand_dims(random.categorical(subkey, log_p_initial), axis=0)

    # FIXME: this should not be needed, p_initial off?
    # add 100 frames, then remove first 100 to allow system to
    # come to equillibrium
    subkeys = random.split(key, num=num_frames + 100)
    _, zs = jax.lax.scan(
        # return value of the scan function is carry and "y", both of which are
        # the next z in our case -> (sample_next_z(),) * 2
        lambda z, k: (sample_next_z(z, p_transition, k),) * 2,
        init=initial_z,
        xs=subkeys,
    )

    key = random.PRNGKey(seed)
    key, subkey = random.split(key)
    x_trace = sample_x_given_z(zs, mu, mu_bg, sigma, key)

    return x_trace[100:, 0], zs[100:]


def sample_next_z(z, p_transition, key):
    p_tr = jnp.log(p_transition[z, :])
    z = random.categorical(key, p_tr)

    return z


def create_transition_matrix(y, p_on, p_off):
    """Create a transition matrix for the number of active elements, given that
    elements can randomly turn on and off.

    Args:

        y (int):
            The maximum number of elements that can be on.

        p_on (float):
            The probability for a single element to turn on (if off).

        p_off (float):
            The probability for a single element to turn off (if on).

    Returns:

        A matrix of transition probabilities of shape ``(y + 1, y + 1)``, with
        element ``i, j`` being the probability that the number of active
        elements changes from ``i`` to ``j``.
    """

    # TODO: this can be cached for larger y (max_y)
    comb_matrix = create_comb_matrix(y)
    comb_matrix_slanted = create_comb_matrix(y, slanted=True)

    # the largest y for which the comb_matrix was generated
    max_y = comb_matrix.shape[0] - 1

    prob_matrix_on = create_prob_matrix(y, p_on, slanted=True)
    prob_matrix_off = create_prob_matrix(y, p_off)

    t_on_matrix = comb_matrix_slanted * prob_matrix_on
    t_off_matrix = comb_matrix * prob_matrix_off

    def correlate(t_on_matrix, t_off_matrix):
        return jax.vmap(lambda a, b: jnp.correlate(a, b, mode="valid"))(
            t_on_matrix[::-1], t_off_matrix
        )

    return correlate(t_on_matrix[: y + 1, max_y - y :], t_off_matrix[: y + 1])


def create_comb_matrix(y, slanted=False):
    """Creates a matrix of n-choose-k values.

    Args:

        y (int):
            The maximum number of elements. The returned matrix will have shape
            ``(y + 1, y + 1)``.

        slanted (bool):
            If given, the returned matrix will be "slanted" to the right, i.e.,
            the second last row will be shifted by 1, the third last one by 2,
            and so on. The shape of the returned matrix will then be ``(y + 1,
            2 * y + 1)``. The slanted form is used to facilitate computation of
            a square transition matrix.

    Returns:

        A matrix of n-choose-k values of shape ``(y + 1, y + 1)``, such that
        the element at position ``i, j`` is the number of ways to select ``j``
        elements from ``i`` elements.
    """

    end_i = y + 1
    end_j = y + 1 if not slanted else 2 * y + 1

    if slanted:
        return jnp.array(
            [[comb(i, j - (y - i)) for j in range(end_j)] for i in range(end_i)]
        )
    else:
        return jnp.array([[comb(i, j) for j in range(end_j)] for i in range(end_i)])


def create_prob_matrix(y, p, slanted=False):
    """Creates a matrix of probabilities for flipping ``i`` out of ``j``
    elements, given that the probability for a single flip is ``p``.

    Args:

        y (int):
            The maximum number of elements. The returned matrix will have shape
            ``(y + 1, y + 1)``.

        p (float):
            The probability of a single flip.

        slanted (bool):
            If given, the returned matrix will be "slanted" to the right, i.e.,
            the second last row will be shifted by 1, the third last one by 2,
            and so on. The shape of the returned matrix will then be ``(y + 1,
            2 * y + 1)``. The slanted form is used to facilitate computation of
            a square transition matrix.

    Returns:

        A matrix of probabilities of shape ``(y + 1, y + 1)``, such that the
        element at position ``i, j`` is the probability to flip ``j`` elements
        out of ``i`` elements, if the probability for a single flip is ``p``.
    """

    i_indices = jnp.arange(0, y + 1)
    j_indices = jnp.arange(0, 2 * y + 1) if slanted else jnp.arange(0, y + 1)

    def prob_i_j(i, j):
        # i are on, j flip
        # -> i - j stay
        # -> j flip

        a = jnp.clip(j, a_min=0.0)
        b = jnp.clip(i - j, a_min=0.0)
        return p**a * (1.0 - p) ** b

    def prob_i(i):
        if slanted:

            def prob_i_fun(j):
                return prob_i_j(i, j - (y - i))

        else:

            def prob_i_fun(j):
                return prob_i_j(i, j)

        return jax.vmap(prob_i_fun)(j_indices)

    return jax.vmap(prob_i)(i_indices)
