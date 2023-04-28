import jax
import jax.numpy as jnp

from .hyper_parameters import HyperParameters
from .parameter_ranges import ParameterRanges
from .parameters import Parameters
# FIXME: post_process should be renamed and find a new home
from .post_process import post_process as find_most_likely_y
from .utils import find_local_maxima
from .trace_model import get_trace_log_likelihood
from .optimizer import create_optimizer


def estimate_y(
        traces,
        max_y,
        parameter_ranges=None,
        hyper_parameters=None):
    """Infer the most likely number of fluorophores for the given traces.

    Args:

        traces (tensor of shape `(n, t)`):

            A list of `n` intensity traces over time.

        max_y (int):

            The maximal `y` (number of fluorophores) to consider.

        parameter_ranges (:class:`ParameterRanges`, optional):

            The parameter ranges to consider for the fluorescence and trace
            model.

        hyper_parameters (:class:`HyperParameters`, optional):

            The hyper-parameters used for the maximum likelihood estimation.

    Returns:

        A tuple `(max_likelihood_y, parameters, log_likelihoods)`.
        `max_likelihood_y` contains the maximum log likelihood solution for
        each trace (shape `(n,)`). `parameters` contains the optimal set of
        fluorescence and trace model parameters for each trace and possible y
        (shape `(n, m, k)`, where `m` is the number of possible ys considered
        and `k` the number of parameters. `log_likelihoods` contains the
        maximum log likelihood for each trace and y (shape `(n, m)`).
    """

    # use defaults if not given

    if parameter_ranges is None:
        parameter_ranges = ParameterRanges()
    if hyper_parameters is None:
        hyper_parameters = HyperParameters()

    # use the maximum intensity in any trace, if not already set

    if hyper_parameters.max_x is None:
        hyper_parameters.max_x = traces.max()

    # fit model for each y separately

    all_parameters = []
    all_log_likelihoods = []
    for y in range(hyper_parameters.min_y, max_y + 1):

        parameters, log_likelihoods = estimate_parameters(
            traces,
            y,
            parameter_ranges,
            hyper_parameters)

        all_parameters.append(parameters)
        all_log_likelihoods.append(log_likelihoods)

    all_parameters = jnp.array(all_parameters)
    all_log_likelihoods = jnp.array(all_log_likelihoods)

    max_likelihood_y = find_most_likely_y(
        traces,
        all_parameters,
        all_log_likelihoods,
        hyper_parameters)

    return max_likelihood_y, all_parameters, all_log_likelihoods


def estimate_parameters(
        traces,
        y,
        parameter_ranges,
        hyper_parameters):
    """Fit the fluorescence and trace model to the given traces, assuming that
    `y` fluorophores are present in each trace.

    Args:

        traces (tensor of shape `(n, t)`):

            A list of `n` intensity traces over time.

        y (int):

            The number of fluorophores to consider.

        parameter_ranges (:class:`ParameterRanges`, optional):

            The parameter ranges to consider for the fluorescence and trace
            model.

        hyper_parameters (:class:`HyperParameters`, optional):

            The hyper-parameters used for the maximum likelihood estimation.

    Returns:

        A tuple `(parameters, log_likelihoods)`. `parameters` contains the
        optimal set of fluorescence and trace model parameters for each trace
        (shape `(n, k)`, where `k` is the number of parameters.
        `log_likelihoods` contains the maximum log likelihood for each trace
        (shape `(n,)`).
    """

    # traces: (n, t)
    # parameter_guesses: (n, g, k)
    # optimizer_states: (n, g, ...)
    # parameters: (n, g, k)
    # log_likelihoods: (n, g)
    # is_done: (n, g)
    #
    # t = length of trace
    # n = number of traces
    # g = number of guesses
    # k = number of parameters

    # get initial guesses for each trace, given the parameter ranges

    parameter_guesses = get_initial_parameter_guesses(
        traces,
        y,
        parameter_ranges,
        hyper_parameters)

    # create the objective function for the given y, as well as its gradient
    # function

    log_likelihood_grad_func = jax.value_and_grad(
        lambda t, p: get_trace_log_likelihood(t, y, p, hyper_parameters),
        argnums=1)

    # create an optimizer, which will be shared between all optimizations

    optimizer = create_optimizer(log_likelihood_grad_func, hyper_parameters)

    # create optimizer states for each trace and parameter guess

    optimizer_states = jax.vmap(jax.vmap(optimizer.init))(parameter_guesses)

    # optimize each trace and parameter guess in parallel until all of them
    # converged

    # fit_epoch(traces, parameters, optimizer_states)
    fit_epoch = jax.vmap(  # vmap over traces
        jax.vmap(  # vmap over parameters
            lambda t, p, os: optimize_parameters(
                t,
                p,
                os,
                optimizer,
                hyper_parameters),
            in_axes=(None, 0, 0)
        )
    )

    # initial conditions
    num_traces = traces.shape[0]
    num_guesses = hyper_parameters.num_guesses
    parameters = parameter_guesses
    is_done = jnp.zeros((num_traces, num_guesses), dtype='bool')

    while not jnp.all(is_done):

        # optimize each trace and guess in parallel for one epoch

        # FIXME: this does not remember log likelihoods from earlier epochs
        parameters, optimizer_states, log_likelihoods, is_done = fit_epoch(
                traces,
                parameters,
                optimizer_states)

        print(f'fitting y = {y}')

        print("log likelihoods:")
        print(log_likelihoods)

        print("is_done:")
        print(is_done)

    # for each trace, keep the best parameter/log likelihood

    best_guesses = jnp.argmin(log_likelihoods, axis=1)

    best_parameters = [
        Parameters(*(
            p[t, best_guesses[t]] for p in parameters
        ))
        for t in range(num_traces)
    ]
    best_log_likelihoods = jnp.array([
        log_likelihoods[t, i]
        for t, i in enumerate(best_guesses)
    ])

    return best_parameters, best_log_likelihoods


def get_initial_parameter_guesses(
        traces,
        y,
        parameter_ranges,
        hyper_parameters):

    '''
    Find rough estimates of the parameters to fit a given trace

    Returns: array of parameters of size 5 x num guesses

    '''
    num_traces = traces.shape[0]
    num_guesses = hyper_parameters.num_guesses

    parameters = parameter_ranges.to_parameters()

    # vmap over parameters
    log_likelihood_over_parameters = jax.vmap(
        lambda t, p: get_trace_log_likelihood(t, y, p, hyper_parameters),
        in_axes=(None, 0)
    )

    # vmap over traces
    log_likelihoods = jax.vmap(
        log_likelihood_over_parameters,
        in_axes=(0, None))(
            traces,
            parameters)

    # reshape parameters so they are "continuous" along each dimension
    parameters = Parameters(*(
        p.reshape(parameter_ranges.num_values())
        for p in parameters
    ))

    # The following calls into non-JAX code and should therefore avoid vmap (or
    # any other transformation like jit or grad). That's why we use a for loop
    # to loop over traces instead of a vmap.
    guesses = []
    for i in range(num_traces):

        # reshape likelihodds to line up with parameters (so they are
        # "continuous" along each dimension)
        trace_log_likelihoods = log_likelihoods[i].reshape(parameter_ranges.num_values())

        # find locations where parameters maximize log likelihoods
        min_indices = find_local_maxima(trace_log_likelihoods, num_guesses)

        guesses.append(Parameters(*(p[min_indices] for p in parameters)))

    # all guesses are stored in 'guesses', the following stacks them together
    # as if we vmap'ed over traces:

    guesses = Parameters(*(
        jnp.stack([guesses[i][p] for i in range(num_traces)])
        for p in range(len(parameters))
    ))

    return guesses


def optimize_parameters(
        trace,
        parameters,
        optimizer_state,
        optimizer,
        hyper_parameters):
    """Fit a single trace and parameter pair, using the given optimizer.

    Returns:

        A tuple `(parameters, optimizer_state, log_likelihood, is_done)`
    """

    # call optimizer.step() num_iterations times, collect all log likelihoods
    # along the way

    # the following is a little helper to do that with jax.scan:
    def step(carry, _):
        parameters, optimizer_state = carry
        parameters, log_likelihood, optimizer_state = optimizer.step(
            trace,
            parameters,
            optimizer_state)
        return (parameters, optimizer_state), log_likelihood

    (parameters, optimizer_state), log_likelihoods = jax.lax.scan(
        step,
        (parameters, optimizer_state),  # carry init
        [],  # x to scan over (nothing in our case)
        length=hyper_parameters.epoch_length)  # number of scan steps

    # mark as done if the most recent log likelihoods do not differ by a lot
    is_done = check_if_converged(
        log_likelihoods,
        hyper_parameters.is_done_limit)

    return parameters, optimizer_state, log_likelihoods[-1], is_done


def check_if_converged(log_likelihoods, limit):
    '''
    Input: an array of log likelihoods shape epoch_length

    output: bool
    '''

    # option_1
    # measures average percent change over last few cycles

    most_recent = log_likelihoods[-10:]
    mean_values = jnp.mean(most_recent)
    mean_improvements = jnp.mean(jnp.diff(most_recent))

    percent_improve = jnp.divide(mean_improvements, mean_values)

    done_improve = percent_improve < limit

    # Check if nan and return true if so
    is_nan = jnp.isnan(percent_improve)
    is_done = jnp.logical_or(done_improve, is_nan)

    return is_done
