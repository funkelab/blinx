from . import transition_matrix
from .constants import (
    PARAM_MU,
    PARAM_MU_BG,
    PARAM_SIGMA,
    PARAM_P_ON,
    PARAM_P_OFF)
from .fluorescence_model import FluorescenceModel
from .hyper_parameters import HyperParameters
from .optimizer import create_optimizer
from .parameter_ranges import ParameterRanges
from .trace_model import TraceModel
from .post_process import post_process
from jax import lax
import jax
import jax.numpy as jnp
import logging
import numpy as np

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

    most_likely_ys, _ = post_process(
        traces=traces,
        parameters=all_parameters,
        likelihoods=all_likelihoods,
        hyper_parameters=hyper_parameters)

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
            hyper_parameters),
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

        print(f'fitting y = {y}')

        print("likelihoods:")
        print(likelihoods)

        print("is_done:")
        print(is_done)

    # for each trace, keep the best parameter/likelihood

    best_guesses = jnp.argmin(likelihoods, axis=1)

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
        hyperparameters):
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
        length=hyperparameters.epoch_length)  # number of scan steps

    # mark as done if the most recent likelihoods do not differ by a lot
    is_done = _is_done(likelihoods, hyperparameters.is_done_limit)

    return parameters, optimizer_state, likelihoods[-1], is_done


def _is_done(likelihoods, limit):
    '''
    Input: an array of likelihoods shape epoch_length

    output: bool
    '''

    # option_1
    # measures average percent change over last few cycles

    most_recent = likelihoods[-10:]
    mean_values = jnp.mean(most_recent)
    mean_improvements = jnp.mean(jnp.diff(most_recent))

    percent_improve = jnp.divide(mean_improvements, mean_values)

    done_improve = percent_improve < limit

    # Check if nan and return true if so
    is_nan = jnp.isnan(percent_improve)
    is_done = jnp.logical_or(done_improve, is_nan)

    return is_done


def get_initial_guesses(y, trace, parameter_ranges, num_guesses):

    '''
    Find rough estimates of the parameters to fit a given trace

    Returns: array of parameters of size 5 x num guesses

    '''
    parameters = parameter_ranges.to_tensor()

    # calculate likelihood for each combination of parameters
    likelihoods = jax.vmap(get_likelihood, in_axes=(None, None, 0))(
        y,
        trace,
        parameters)

    # reshape parameters and likelihoods so they are "continuous"
    # along each dimension
    parameters = parameters.reshape(
        parameter_ranges.num_values() +
        (len(parameter_ranges.num_values()),)
    )
    results = likelihoods.reshape(
        parameter_ranges.num_values())

    # find locations where parameters minimize likelihoods
    min_c = _find_minima_nd(results, num_guesses)

    return parameters[min_c[:, 0], min_c[:, 1], min_c[:, 2], min_c[:, 3],
                      min_c[:, 4]]


def _minima_point(index, b, a):
    # Given a coordinate, finds the nearest neighbors and determines if given
    # coordinate is a local minima
    centered = b - b[index]
    dist = jnp.linalg.norm(centered, axis=1)
    c = jnp.where(dist <= 1, size=len(a.shape*2)+1)
    tile = b[c]
    tile_values = a[tile[:, 0], tile[:, 1]]
    all_same = jnp.all(tile_values == tile_values[0])
    result = jax.lax.cond(all_same, lambda: 0., lambda: jnp.min(tile_values))
    return result


def _find_minima_nd(matrix, num_minima):
    '''
    Find local minima of an N dimensional matrix
    - finds nearest neighbors of each point
    - compares neighbors to determine if point is a minima

    Returns:
        a 5 x num_minima array containing the coordinates of the found
        local minima
    '''

    # FIXME: has to be a way to avoid hard coding this
    # but not the worst because input params will always be 5d
    shape = matrix.shape
    dim_1 = np.arange(shape[0])
    dim_2 = np.arange(shape[1])
    dim_3 = np.arange(shape[2])
    dim_4 = np.arange(shape[3])
    dim_5 = np.arange(shape[4])

    b = jnp.asarray(np.meshgrid(
        dim_1, dim_2, dim_3, dim_4, dim_5
        )).reshape((matrix.ndim, np.product(matrix.shape))).T

    indices = jnp.arange(np.product(matrix.shape))
    d = jax.vmap(_minima_point, in_axes=(0, None, None))(indices, b, matrix)

    e = matrix.reshape(np.product(matrix.shape))

    return b[jnp.where(d == e, size=num_minima)]


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
        sigma=sigma,
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
    # FIXME: invert gradients in optimizer instead
    return -1 * likelihood
