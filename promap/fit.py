from . import transition_matrix
from .constants import (
    PARAM_MU,
    PARAM_MU_BG,
    PARAM_SIGMA,
    PARAM_P_ON,
    PARAM_P_OFF)
from .constants import P_ON, P_OFF, MU, SIGMA
from .fluorescence_model import FluorescenceModel
from .hyper_parameters import HyperParameters
from .optimizer import create_optimizer
from .parameter_ranges import ParameterRanges
from .trace_model import TraceModel
from jax import lax
import jax
import jax.numpy as jnp
import logging
import numpy as np
import optax

logger = logging.getLogger(__name__)


def most_likely_ys(
        traces,
        y_low,
        y_high,
        parameter_ranges=None,
        hyper_parameters=None):
    """Infer the most likely number of fluorophores for the given traces.

    Args:

        traces (tensor of shape `(n, t)`):

            A list of `n` intensity traces over time.

        y_low, y_high (int):

            The minimal and maximal `y` (number of fluorophores) to consider.

        parameter_ranges (:class:`ParameterRanges`, optional):

            The parameter ranges to consider for the fluorescence and trace
            model.

        hyper_parameters (:class:`HyperParameters`, optional):

            The hyperparameters used for the maximum likelihood estimation.

    Returns:

        A tuple `(most_likely_ys, parameters, likelihoods)`. `most_likely_ys`
        contains the maximum likelihood solution for each trace (shape `(n,)`).
        `parameters` contains the optimal set of fluorescence and trace model
        parameters for each trace and y (shape `(n, m, k)`, where `m` is the
        number of ys considered and `k` the number of parameters. `likelihoods`
        contains the maximum likelihood for each trace and y (shape `(n, m)`).
    """

    # use defaults if not given

    if parameter_ranges is None:
        parameter_ranges = ParameterRanges()
    if hyper_parameters is None:
        hyper_parameters = HyperParameters()

    # fit model for each y separately

    all_parameters = []
    all_likelihoods = []
    for y in range(y_low, y_high + 1):

        parameters, likelihoods = fit_traces(
            y,
            traces,
            parameter_ranges,
            hyper_parameters)

        all_parameters.append(parameters)
        all_likelihoods.append(likelihoods)

    all_parameters = jnp.array(all_parameters)
    all_likelihoods = jnp.array(all_likelihoods)

    most_likely_ys = jnp.argmax(all_likelihoods, axis=1) + y_low

    return most_likely_ys, all_parameters, all_likelihoods


def fit_traces(
        y,
        traces,
        parameter_ranges,
        hyper_parameters):
    """Fit the fluorescence and trace model to the given traces, assuming that
    `y` fluorophores are present in each trace.

    Args:

        y (int):

            The number of fluorophores to consider.

        traces (tensor of shape `(n, t)`):

            A list of `n` intensity traces over time.

        parameter_ranges (:class:`ParameterRanges`, optional):

            The parameter ranges to consider for the fluorescence and trace
            model.

        hyper_parameters (:class:`HyperParameters`, optional):

            The hyperparameters used for the maximum likelihood estimation.

    Returns:

        A tuple `(parameters, likelihoods)`. `parameters` contains the optimal
        set of fluorescence and trace model parameters for each trace (shape
        `(n, k)`, where `k` is the number of parameters. `likelihoods` contains
        the maximum likelihood for each trace (shape `(n,)`).
    """

    # traces: (n, t)
    # parameter_guesses: (n, g, k)
    # optimizer_states: (n, g, ...)
    # parameters: (n, g, k)
    # likelihoods: (n, g)
    # is_done: (n, g)
    #
    # t = length of trace
    # n = number of traces
    # g = number of guesses
    # k = number of parameters

    # get initial guesses for each trace, given the parameter ranges

    parameter_guesses = jax.vmap(
        get_initial_guesses,
        in_axes=(None, 0, None, None))(
            y,
            traces,
            parameter_ranges,
            hyper_parameters.num_guesses)

    num_traces = parameter_guesses.shape[0]
    num_guesses = parameter_guesses.shape[1]

    # create the objective function for the given y, as well as its gradient
    # function

    likelihood_grad_func = jax.value_and_grad(
        lambda t, p: get_likelihood(y, t, p),
        argnums=1)

    # create an optimizer, which will be shared between all optimizations

    optimizer = create_optimizer(likelihood_grad_func, hyper_parameters)

    # create optimizer states for each trace and parameter guess

    optimizer_states = jax.vmap(jax.vmap(optimizer.init))(parameter_guesses)

    # optimize each trace and parameter guess in parallel until all of them
    # converged
    #
    # we do this with two nested vmaps:
    #
    #   vmap_parameters(trace, parameters, optimizer_states)
    #
    # and
    #
    #   vmap_traces(traces, parameters, optimizer_states)

    # vmap over parameter guesses and their corresponding optimizers
    vmap_parameters = jax.vmap(
        # just fit_trace, but with optimizer and epoch_length bound
        lambda t, p, os: fit_trace(
            t,
            p,
            os,
            optimizer,
            hyper_parameters.epoch_length),
        in_axes=(None, 0, 0))
    # vmap over traces
    vmap_traces = jax.vmap(vmap_parameters, in_axes=(0, 0, 0))

    # final epoch fit function
    fit_epoch = jax.jit(vmap_traces)

    # initial conditions
    parameters = parameter_guesses
    is_done = jnp.zeros((num_traces, num_guesses), dtype='bool')

    while not jnp.all(is_done):

        # optimize each trace and guess in parallel for one epoch

        parameters, optimizer_states, likelihoods, is_done = fit_epoch(
                traces,
                parameters,
                optimizer_states)

        print("likelihoods:")
        print(likelihoods)

        print("is_done:")
        print(is_done)

    # for each trace, keep the best parameter/likelihood

    best_guesses = jnp.argmax(likelihoods, axis=1)

    best_parameters = jnp.array([
        parameters[t, i]
        for t, i in enumerate(best_guesses)
    ])
    best_likelihoods = jnp.array([
        likelihoods[t, i]
        for t, i in enumerate(best_guesses)
    ])

    return best_parameters, best_likelihoods


def fit_trace(
        trace,
        parameters,
        optimizer_state,
        optimizer,
        num_iterations):
    """Fit a single trace and parameter pair, using the given optimizer.

    Returns:

        A tuple `(parameters, optimizer_state, likelihood, is_done)`
    """

    # call optimizer.step() num_iterations times, collect all likelihoods along
    # the way

    # the following is a little helper to do that with jax.scan:
    def step(carry, _):
        parameters, optimizer_state = carry
        parameters, likelihood, optimizer_state = optimizer.step(
            trace,
            parameters,
            optimizer_state)
        return (parameters, optimizer_state), likelihood

    (parameters, optimizer_state), likelihoods = lax.scan(
        step,
        (parameters, optimizer_state),  # carry init
        [],  # x to scan over (nothing in our case)
        length=num_iterations)  # number of scan steps

    # mark as done if the most recent likelihoods do not differ by a lot

    is_done = jnp.abs(likelihoods[-1] - likelihoods[-2]) < 1e-4

    return parameters, optimizer_state, likelihoods[-1], is_done


def get_initial_guesses(y, trace, parameter_ranges, num_guesses):

    # TODO: this is just a dummy implementation that simply returns the first
    # num_guesses parameters

    return parameter_ranges.to_tensor()[:num_guesses]


def get_likelihood(y, trace, parameters):
    '''
    Returns the likelihood of a trace given:
        a count (y),
        kinetic parameters (p_on & p_off), and
        emission parameters (mu & sigma)
    '''

    mu = parameters[PARAM_MU]
    mu_bg = parameters[PARAM_MU_BG]
    sigma = parameters[PARAM_SIGMA]
    p_on = parameters[PARAM_P_ON]
    p_off = parameters[PARAM_P_OFF]

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

# ---------------------------------------------------------------
# FIXME: functions above replace the functions below
#        -> remove once the above is tested and working
# ---------------------------------------------------------------


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
    all_params = np.zeros((len(y_range), 6))

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

    def bound_likelihood(index, p_on, p_off, mu, sigma):
        return index_likelihood_func(
            index, y, p_on, p_off, mu, sigma, trace,
            mu_b_guess=mu_b_guess)

    grad_func = jax.jit(
        jax.value_and_grad(bound_likelihood, argnums=(1, 2, 3, 4)))

    if initial_params is None:
        initial_params = _initial_guesses(
            mu_min=100, p_max=0.2, y=y, trace=trace,
            mu_b_guess=mu_b_guess, sigma=sigma_guess)

    p_ons = initial_params[P_ON]
    p_offs = initial_params[P_OFF]
    mus = initial_params[MU]
    sigmas = initial_params[SIGMA]

    likelihood, p_on, p_off, mu, sigma = optimize(
        p_ons,
        p_offs,
        mus,
        sigmas,
        mu_lr,
        grad_func)

    return likelihood, [p_on, p_off, mu, sigma, y, likelihood]


def optimize(p_ons, p_offs, mus, sigmas, mu_lr, grad_func):

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

    def bound_likelihood(mu, p_on, p_off):
        return _likelihood_func(
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
