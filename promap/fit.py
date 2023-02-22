import jax.numpy as jnp
import numpy as np
import jax
from jax import lax
from promap import transition_matrix
from .trace_model import TraceModel
from .fluorescence_model import FluorescenceModel
from .parameter_ranges import ParameterRanges
from .constants import MU, MU_BG, SIGMA, P_ON, P_OFF
import optax
import logging

logger = logging.getLogger(__name__)


def most_likely_y(
        trace,
        y_low,
        y_high,
        parameter_ranges=None):

    if parameter_ranges is None:
        parameter_ranges = ParameterRanges()

    y_range = np.arange(y_low, y_high+1)
    logger.info("Finding most likely y in %s", list(y_range))

    likelihoods = np.zeros((len(y_range)))
    all_params = np.zeros((len(y_range), 6))

    for i, y in enumerate(y_range):
        likelihood, params = optimize_params(
            y,
            trace=trace,
            parameter_ranges=parameter_ranges)
        likelihoods[i] = likelihood
        all_params[i, :] = params
        logger.info("y=%d    likelihood=%.2f", y, likelihood)
    most_likely_y = y_range[np.argmax(likelihoods)]

    return most_likely_y, all_params, likelihoods, list(y_range)


def optimize_params(
        y,
        trace,
        parameter_ranges,
        mu_lr=5):
    '''
    Fit kinetic (p_on / off) and emission (mu / sigma) parameters
    to an intensity trace for a given value of y

    Args:
        y (int):
            - The assumed total number of fluorescent emitters

        trace (jnp array):
            - ordered array of intensity observations
            - shape (number_observations, )

        parameter_ranges (:class:`ParameterRanges`):
            - initial guesses for p_on, p_off, mu, mu_bg, and sigma

        mu_lr (float):
            - the learning rate for the mu optimizer
            - needs to be individually set because of difference in magnitude
            between mu and other parameters

    Returns:
        The maximum log-likelihood that the trace arrose from y elements,
        as well as the optimum values of p_on, p_off, mu, and sigma
    '''

    # create the likelihood and gradient function for specific parameter
    # values, but don't compute the gradient wrt. mu_bg, we do not need it
    grad_func = jax.jit(
        jax.value_and_grad(
            _likelihood_func,
            argnums=(2,)))

    parameters = initial_guesses(y, trace, parameter_ranges)

    likelihood, best_parameters = fit_all(
        y,
        jnp.array([trace]),
        parameters,
        mu_lr,
        grad_func)

    return likelihood, best_parameters


def fit_single(y, trace, parameters, num_iterations, optimizer, opt_state):

    def step(parameters, opt_state):

        # TODO: pull out
        grad_func = jax.value_and_grad(
            _likelihood_func,
            argnums=(2,))(
                y,
                trace,
                parameters)

        likelihood, gradients = grad_func(y, trace, parameters)
        updates, opt_state = optimizer.update(gradients, opt_state)

        parameters = optax.apply_updates(parameters, updates)

        return (parameters, opt_state), likelihood

    (parameters, opt_state), likelihoods = lax.scan(
        step,
        (parameters, opt_state),
        [],
        length=num_iterations)

    is_done = jnp.abs(likelihoods[-1] - likelihoods[-2]) < 1e-4

    return is_done, parameters, opt_state


def fit_trace_epoch(y, trace, parameters, epoch_length, optimizers, opt_states):

        is_done, parameters = vmap(
                fit_single,
                in_axes=(None, None, 0, None, 0, 0)(
            y,
            trace,
            parameters,
            epoch_length,
            optimizers,
            opt_states)

        return is_done, parameters


def fit_all(y, traces, parameter_ranges, epoch_length):

    # a short list of parameters to explore per trace
    # parameters: (num_traces, num_guesses, num_parameters)
    # TODO:
    #   * initial_guesses should always return "num_guesses" parameters
    #   * all returned values need to be copies of the parameters
    parameters = vmap(initial_guesses, in_axes=(None, 0, None))(
        y,
        trace,
        parameter_ranges)

    # input  : (y, traces, parameters, epoch_length, optimizers, opt_states)
    # returns:
    #       is_done (num_traces, num_guesses),
    #       parameters (num_traces, num_guesses, num_parameters)
    fit_traces_vmap = vmap(
        fit_trace_epoch,
        in_axes=(None, 0, 0, None, None, None))

    # value: (y, traces, parameters, epoch_length, optimizers, opt_states, is_done)

    def is_done_condition(val):
        is_done = val[-1]
        return jnp.all(is_done)

    def fit_traces(val):
        args = val[:-1]
        return fit_traces_vmap(*args)

    is_done = jnp.zeros((num_traces, num_configs), dtype=jnp.bool)
    val = lax.while_loop(
        is_done_condition,
        fit_traces,
        (y, traces, parameters, epoch_length, optimizers, opt_states, is_done))

    return val[2]  # parameters


def optimize(y, trace, parameters, mu_lr, grad_func):

    mus = parameters[:, MU]
    mu_bgs = parameters[:, MU_BG]
    sigmas = parameters[:, SIGMA]
    p_ons = parameters[:, P_ON]
    p_offs = parameters[:, P_OFF]

    indices = jnp.arange(len(p_ons))

    params = (p_ons, p_offs, mus, sigmas)
    optimizer = optax.adam(learning_rate=1e-3, mu_dtype='uint64')
    opt_state = optimizer.init(params)

    mu_optimizer = optax.sgd(learning_rate=mu_lr)
    mu_opt_state = mu_optimizer.init(params[2])

    old_likelihoods, _ = jax.vmap(
        grad_func, in_axes=(0, None, None, None, None))(
        indices, mus, mu_bgs, sigmas, p_ons, p_offs)

    old_likelihoods += 1
    diff = -10

    while diff < -1e-4:
        likelihoods, grads = jax.vmap(
            grad_func, in_axes=(0, None, None, None, None))(
            indices, mus, mu_bgs, sigmas, p_ons, p_offs)

        updates, opt_state = optimizer.update(grads, opt_state)

        mu_update, mu_opt_state = mu_optimizer.update(grads[2], mu_opt_state)

        p_ons, p_offs, _, sigmas = optax.apply_updates((
            p_ons, p_offs, mus, sigmas), updates)

        mus = optax.apply_updates((mus), mu_update)

        p_ons = p_ons[indices, indices]
        p_offs = p_offs[indices, indices]
        mus = mus[indices, indices]
        sigmas = sigmas[indices, indices]

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


def _likelihood_func(y, trace, parameters):
    '''
    Returns the likelihood of a trace given:
        a count (y),
        kinetic parameters (p_on & p_off), and
        emission parameters (mu & sigma)
    '''

    mu = parameters[MU]
    mu_bg = parameters[MU_BG]
    sigma = parameters[SIGMA]
    p_on = parameters[P_ON]
    p_off = parameters[P_OFF]

    fluorescence_model = FluorescenceModel(
        mu_i=mu,
        sigma_i=sigma,
        mu_b=mu_bg)
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


def initial_guesses(y, trace, parameter_ranges):
    '''
    Provides a rough estimate of parameters
    Grid searches over defined parameter space and returns the minimum
    log likelihood parameters
    '''

    logger.debug("Finding initial guesses for y=%d...", y)

    # get a tensor of all possible parameter combinations
    parameters = parameter_ranges.to_tensor()

    # compute likelihood of each
    results = jax.vmap(_likelihood_func, in_axes=(None, None, 0))(
        y,
        trace,
        parameters)

    # reshape parameters and likelihood array to be congruent with parameter
    # space
    num_parameter_values = parameter_ranges.num_values()
    parameters = parameters.reshape(
        num_parameter_values +
        (len(num_parameter_values),)
    )
    results = results.reshape(num_parameter_values)

    # find local minima in parameter space
    #
    # FIXME: this works in 3D only for now, we use this to find the best in
    # (mu, p_on, p_off) only
    # -> this assumes that parameters are (mu, mu_bg, sigma, p_on, p_off)
    # FIXME: also assumes that there is only one value for mu_bg and sigma
    # FIXME: handle case if no minima were found
    minima_indices = _find_minima_3d(results[:, 0, 0, :, :], 3)

    print("results:")
    print(results[:, 0, 0, :, :])

    print("minima indices:")
    print(minima_indices)

    # select only minima parameter values
    return parameters[
        minima_indices[MU],
        :,
        :,
        minima_indices[P_ON],
        minima_indices[P_OFF]
    ].reshape(-1, len(num_parameter_values))


def _find_minima_3d(test_vec, window):
    '''
    - Finds the local minima of a 3D array
    - returns the minima indices as an array of shape 3 x num_minima
    - for the first axis: 0 = p_on, 1 = p_off, 2 = mu
    '''
    mu_indices = jnp.arange(test_vec.shape[0]+window)[window:]
    p_off_indices = jnp.arange(test_vec.shape[1]+window)[window:]
    p_on_indices = jnp.arange(test_vec.shape[2]+window)[window:]

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
        in_axes=(None, 0, None, None))(test_vec_pad, mu_indices,
                                       p_on_indices, p_off_indices)

    # trim edges because minima at edges cant be local minima
    new_a = jnp.pad(a[1:-1, 1:-1, 1:-1], 1)
    local_minima = jnp.asarray(jnp.where(new_a == test_vec))

    return local_minima
