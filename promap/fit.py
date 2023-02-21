import jax.numpy as jnp
import numpy as np
import jax
from jax import lax
from promap.trace_model import TraceModel
from promap.fluorescence_model import FluorescenceModel
from promap import transition_matrix
from promap.constants import P_ON, P_OFF, MU, SIGMA
import optax
import logging

logger = logging.getLogger(__name__)


def most_likely_y(
        trace,
        y_low,
        y_high,
        mu_b_guess=5000):
    '''


    '''

    y_range = np.arange(y_low, y_high+1)
    logger.info("Finding most likely y in %s", list(y_range))

    likelihoods = np.zeros((len(y_range)))
    all_params = np.zeros((len(y_range), 5))

    for i, y in enumerate(y_range):
        likelihood, params = optimize_params(
            y,
            trace=trace,
            initial_params=None,
            sigma_guess=0.1)
        likelihoods[i] = likelihood
        all_params[i, :] = params
        logger.info("y=%d    likelihood=%.2f", y, likelihood)
    most_likely_y = y_range[np.argmax(likelihoods)]

    return most_likely_y, all_params, likelihoods, list(y_range)


def optimize_params(
        y,
        trace,
        initial_params=None,
        optimize_meth='joint_2_optimizer',
        mu_b_guess=5000,
        mu_lr=5,
        sigma_guess=0.2):
    '''
    Fit kinetic (p_on / off) and emission (mu / sigma) parameters
    to an intensity trace for a given value of y

    Args:
        y (int):
            - The assumed total number of fluorescent emitters

        trace (jnp array):
            - ordered array of intensity observations
            - shape (number_observations, )

        initial_params (list of arrays) (float) or None:
            - initial guesses for p_on, p_off, mu, and sigma
            - format list([P_ON], [P_OFF], [MU], [SIGMA])
            - if None, then will automatically find initial guesses

        optimize_meth (string):
            - specifies the optimizer to use for gradient descent

        mu_b_guess (float / int):
            - guess for background intensity value

        mu_lr (float):
            - the learning rate for the mu optimizer
            - needs to be individually set because of difference in magnitude
            between mu and other parameters

    Returns:
        The maximum log-likelihood that the trace arrose from y elements,
        as well as the optimum values of p_on, p_off, mu, and sigma
    '''

    def index_likelihood_func(
            index, y, p_on, p_off, mu, sigma, trace, mu_b_guess):
        return _likelihood_func(
            y, p_on[index], p_off[index], mu[index],
            sigma[index], trace, mu_b_guess)

    bound_likelihood = lambda index, p_on, p_off, mu, sigma: \
        index_likelihood_func(
            index, y, p_on, p_off, mu, sigma, trace,
            mu_b_guess=mu_b_guess)

    grad_func = jax.jit(
        jax.value_and_grad(bound_likelihood, argnums=(1, 2, 3, 4)))

    if initial_params is None:
        initial_params = _initial_guesses(
            mu_min=100, p_max=0.2, y=y, trace=trace,
            mu_b_guess=mu_b_guess, sigma=sigma_guess)

    if optimize_meth == 'joint_2_optimizer':
        optimizer = _optimizer_1

    p_ons = initial_params[P_ON]
    p_offs = initial_params[P_OFF]
    mus = initial_params[MU]
    sigmas = initial_params[SIGMA]

    likelihood, p_on, p_off, mu, sigma = optimizer(
        p_ons,
        p_offs,
        mus,
        sigmas,
        mu_lr,
        grad_func)

    return likelihood, [p_on, p_off, mu, sigma, y]


def _optimizer_1(p_ons, p_offs, mus, sigmas, mu_lr, grad_func):

    indecies = jnp.arange(len(p_ons))

    params = (p_ons, p_offs, mus, sigmas)
    optimizer = optax.adam(learning_rate=1e-3, mu_dtype='uint64')
    opt_state = optimizer.init(params)

    mu_optimizer = optax.sgd(learning_rate=mu_lr)
    mu_opt_state = mu_optimizer.init(params[2])

    old_likelihoods, _ = jax.vmap(
        grad_func, in_axes=(0, None, None, None, None))(
        indecies, p_ons, p_offs, mus, sigmas)

    old_likelihoods += 1
    diff = -10

    while diff < -1e-4:
        likelihoods, grads = jax.vmap(
            grad_func, in_axes=(0, None, None, None, None))(
            indecies, p_ons, p_offs, mus, sigmas)

        updates, opt_state = optimizer.update(grads, opt_state)

        mu_update, mu_opt_state = mu_optimizer.update(grads[2], mu_opt_state)

        p_ons, p_offs, _, sigmas = optax.apply_updates((
            p_ons, p_offs, mus, sigmas), updates)

        mus = optax.apply_updates((mus), mu_update)

        p_ons = p_ons[indecies, indecies]
        p_offs = p_offs[indecies, indecies]
        mus = mus[indecies, indecies]
        sigmas = sigmas[indecies, indecies]

        diff = jnp.min(likelihoods - old_likelihoods)
        old_likelihoods = likelihoods

        logger.debug(
            "likelihood=%.2f, p_on=%.4f, p_off=%.4f, mu=%.4f, sigma=%.4f",
            likelihoods,
            p_ons,
            p_offs,
            mus,
            sigmas)

    b_index = jnp.argmin(likelihoods)

    return -1*likelihoods[b_index], p_ons[b_index], \
        p_offs[b_index], mus[b_index], sigmas[b_index]


def _likelihood_func(y, p_on, p_off, mu, sigma, trace, mu_b_guess):
    '''
    Returns the likelihood of a trace given:
        a count (y),
        kinetic parameters (p_on & p_off), and
        emission parameters (mu & sigma)
    '''
    fluorescence_model = FluorescenceModel(
        mu_i=mu,
        sigma_i=sigma,
        mu_b=mu_b_guess,
        sigma_b=0.05)
    t_model = TraceModel(fluorescence_model)

    probs = t_model.fluorescence_model.p_x_given_zs(trace, y)

    comb_matrix = transition_matrix._create_comb_matrix(y)
    comb_matrix_slanted = transition_matrix._create_comb_matrix(
        y,
        slanted=True)

    def c_transition_matrix_2(p_on, p_off):
        return transition_matrix.create_transition_matrix(
            y, p_on, p_off,
            comb_matrix,
            comb_matrix_slanted)

    transition_mat = c_transition_matrix_2(p_on, p_off)
    p_initial = transition_matrix.p_initial(y, transition_mat)
    likelihood = t_model.get_likelihood(
        probs,
        transition_mat,
        p_initial)

    # need to flip to positive value for grad descent
    return -1 * likelihood


def _initial_guesses(mu_min, p_max, y, trace, mu_b_guess, sigma=0.05):
    '''
    Provides a rough estimate of parameters (p_on, p_off, and mu)
    Grid searches over defined parameter space and returns the minimum
    log likelihood parameters
    '''

    logger.debug("Finding initial guesses for y=%d...", y)

    mus = jnp.linspace(mu_min, jnp.max(trace), 100)
    p_s = jnp.linspace(1e-4, p_max, 20)

    bound_likelihood = lambda mu, p_on, p_off: _likelihood_func(
        y, p_on, p_off, mu, sigma, trace, mu_b_guess)

    result = jax.vmap(jax.vmap(jax.vmap(
        bound_likelihood,
        in_axes=(0, None, None)),
        in_axes=(None, 0, None)),
        in_axes=(None, None, 0))(mus, p_s, p_s)

    minima_indecies = _find_minima_3d(result, 3)
    if minima_indecies.shape[1] == 0:
        print('no local minima found, using median guess values')
        minima_indecies = np.zeros((3, 1)).astype('uint8')
        minima_indecies[:, 0] = [10, 10, 50]

    p_on_guess = p_s[minima_indecies[P_ON, :]]
    p_off_guess = p_s[minima_indecies[P_OFF, :]]
    mu_guess = mus[minima_indecies[MU, :]]
    sigma_guess = jnp.ones((mu_guess.shape)) * sigma
    likelihoods = result[tuple(minima_indecies)]

    logger.debug(
        "...found p_on=%s, p_off=%s, mu=%s",
        p_on_guess,
        p_off_guess,
        mu_guess)

    return (p_on_guess, p_off_guess, mu_guess, sigma_guess, likelihoods)


def _find_minima_3d(test_vec, window):
    '''
    - Finds the local minima of a 3D array
    - returns the minima indecies as an array of shape 3 x num_minima
    - for the first axis: 0 = p_on, 1 = p_off, 2 = mu
    '''
    mu_indecies = jnp.arange(test_vec.shape[2]+window)[window:]
    p_off_indecies = jnp.arange(test_vec.shape[0]+window)[window:]
    p_on_indecies = jnp.arange(test_vec.shape[0]+window)[window:]

    def scan_func(vector, p_on_index, p_off_index, mu_index):
        vector_slice = lax.dynamic_slice(
            vector,
            (p_on_index-window, p_off_index-window, mu_index-window),
            (2*window, 2*window, 2*window))
        slice_min = jnp.min(vector_slice)
        all_same = jnp.all(vector_slice == vector_slice[0])
        b = jax.lax.cond(all_same, lambda: 0., lambda: slice_min)
        return b

    test_vec_pad = jnp.pad(test_vec, window, mode='maximum')
    a = jax.vmap(jax.vmap(jax.vmap(
        scan_func,
        in_axes=(None, None, None, 0)),
        in_axes=(None, None, 0, None)),
        in_axes=(None, 0, None, None))(test_vec_pad, p_on_indecies,
                                       p_off_indecies, mu_indecies)

    # trim edges because minima at edges cant be local minima
    new_a = jnp.pad(a[1:-1, 1:-1, 1:-1], 1)
    local_minima = jnp.asarray(jnp.where(new_a == test_vec))

    return local_minima
