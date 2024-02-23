import collections

import jax
import jax.numpy as jnp
from tqdm import tqdm

from .hyper_parameters import HyperParameters
from .optimizer import create_adam_optimizer
from .parameter_ranges import ParameterRanges
from .parameters import Parameters

# FIXME: post_process should be renamed and find a new home
from .post_process import post_process as find_most_likely_y
from .trace_model import get_trace_log_likelihood, log_p_x_parameters
from .utils import find_maximum


def estimate_y(
    traces, max_y, parameter_ranges=None, hyper_parameters=None, initial_parameters=None
):
    """Infer the most likely number of fluorophores for the given traces.

    Args:

        traces (tensor of shape `(n, t)`):

            A list of `n` intensity traces over time.

        max_y (int):

            The maximal `y` (number of fluorophores) to consider.

        parameter_ranges (:class:`ParameterRanges`, optional):

            The fitting bounds for each parameter to be optimized

        hyper_parameters (:class:`HyperParameters`, optional):

            The hyper-parameters used for the maximum likelihood estimation.

    Returns:

        max_likelihood_y (array):

            the maximum log likelihood solution for
            each trace (shape `(n,)`)

        parameters (:class:`Parameters`):

            the optimal set of fluorescence and kinetic model parameters for each
            trace and possible y (shape `(n, m, k)`), where `m` is the number
            of ys considered and `k` the number of parameters

        log_likelihoods (array):
            the maximum log likelihood for each trace and y (shape `(n, m)`)
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
    all_log_evidences = []
    for y in range(hyper_parameters.min_y, max_y + 1):
        parameters, log_likelihoods, log_evidence = estimate_parameters(
            traces, y, parameter_ranges, hyper_parameters, initial_parameters
        )

        all_parameters.append(parameters)
        all_log_likelihoods.append(log_likelihoods)
        all_log_evidences.append(log_evidence)

    all_parameters = Parameters.stack(all_parameters)
    all_log_likelihoods = jnp.array(all_log_likelihoods)
    all_log_evidences = jnp.array(all_log_evidences)

    max_likelihood_y = find_most_likely_y(
        traces, all_parameters, all_log_likelihoods, hyper_parameters
    )

    return max_likelihood_y[0], all_parameters, all_log_likelihoods, all_log_evidences


def estimate_parameters(
    traces, y, parameter_ranges, hyper_parameters, initial_parameters
):
    """Fit the fluorescence and trace model to the given traces, assuming that
    `y` fluorophores are present in each trace.

    Args:

        traces (tensor of shape `(n, t)`):

            A list of `n` intensity traces over time.

        y (int):

            The number of fluorophores to consider.

        parameter_ranges (:class:`ParameterRanges`):

            The fitting bounds for parameter optimization

        hyper_parameters (:class:`HyperParameters`):

            The hyper-parameters used for the maximum likelihood estimation.

        initial_parameters (:class: `Parameters`):

            Initial guesses for the parameters, if None guess them from a grid search over parameter_ranges

    Returns:

        parameters (:class:`Parameters`):

            the optimal set of fluorescence and kinetic model parameters for each trace
            (shape `(n, k)`), where `k` is the number of parameters.

        log_likelihoods (array):

            the maximum log likelihood for each trace
            (shape `(n,)`)
    """

    # traces: (n, t)
    # parameters: (n, g, k)
    # optimizer_states: (n, g, ...)
    # parameters: (n, g, k)
    # log_likelihoods: (n, g)
    #
    # t = length of trace
    # n = number of traces
    # g = number of guesses
    # k = number of parameters

    # get initial guesses for each trace, given the parameter ranges

    if initial_parameters is None:
        parameters = get_initial_parameter_guesses(
            traces, y, parameter_ranges, hyper_parameters
        )
    else:
        parameters = initial_parameters

    # create the objective function for the given y, as well as its gradient
    # function

    grad_func = jax.value_and_grad(
        lambda t, p, p_loc, p_scale: log_p_x_parameters(
            t, y, p, hyper_parameters, p_loc, p_scale
        ),
        argnums=1,
    )

    hessian_func = jax.hessian(
        lambda t, p, p_loc, p_scale: log_p_x_parameters(
            t, y, p, hyper_parameters, p_loc, p_scale
        ),
        argnums=1,
    )

    # create an optimizer, which will be shared between all optimizations

    optimizer = create_adam_optimizer(grad_func, hyper_parameters)

    # create optimizer states for each trace and parameter guess

    optimizer_states = jax.vmap(jax.vmap(optimizer.init))(parameters)

    vmap_parameters = jax.vmap(optimizer.step, in_axes=(None, 0, 0, None, None))
    vmap_traces = jax.vmap(vmap_parameters)  # in_axes=(0, None, None, 0, 0))
    optimizer_step = jax.jit(vmap_traces)

    log_likelihoods_history = collections.deque(maxlen=hyper_parameters.is_done_window)

    for i in tqdm(range(hyper_parameters.epoch_length), f"y={y}"):
        parameters, log_likelihoods, optimizer_states, gradients = optimizer_step(
            traces,
            parameters,
            optimizer_states,
            hyper_parameters.prior_locs,
            hyper_parameters.prior_scales,
        )

        log_likelihoods_history.append(log_likelihoods)
        if is_done(log_likelihoods_history, hyper_parameters):
            break

    hessian_vmap_parameters = jax.vmap(hessian_func, in_axes=(None, 0, None, None))
    hessian_vmap_traces = jax.vmap(hessian_vmap_parameters)
    # not sure about this negative number (Bishop 4.137)
    occam_factor_a = -hessian_vmap_traces(
        traces, parameters, hyper_parameters.prior_locs, hyper_parameters.prior_scales
    ).flatten()
    # (n, n, t, g)
    occam_factor_a = jnp.transpose(occam_factor_a, axes=(2, 3, 0, 1))
    # (t, g, n, n)

    occam_factors = -0.5 * jnp.log(jnp.linalg.det(occam_factor_a))
    # (t, g)

    # for each trace, keep the best parameter/log likelihood

    best_guesses = jnp.argmin(log_likelihoods, axis=1)

    best_indices = (
        tuple(range(len(best_guesses))),
        tuple(best_guesses),
    )
    best_parameters = parameters[best_indices]
    best_log_likelihoods = log_likelihoods[best_indices]
    best_occam_factors = occam_factors[best_indices]

    best_log_evidence = best_log_likelihoods + best_occam_factors

    print("log likelihoods")
    print(best_log_likelihoods)
    print("-" * 50)
    print("log_evidence")
    print(best_log_evidence)

    return best_parameters, best_log_likelihoods, best_log_evidence


def get_initial_parameter_guesses(traces, y, parameter_ranges, hyper_parameters):
    """
    Find rough estimates of the parameters as starting points for parameter optimization.

    Args:

        traces (tensor of shape `(n, t)`):

            A list of `n` intensity traces over time.

        y (int):

            The number of fluorophores to consider.

        parameter_ranges (:class:`ParameterRanges`):

            The fitting bounds for parameter optimization

        hyper_parameters (:class:`HyperParameters`):

            The hyper-parameters used for the maximum likelihood estimation.

    Returns:

        parameters (:class:`Parameters`):

            The inital parameter guesses for each trace, shape '(n * i, k)' where
            'i' is the number of initial guesses per trace, and 'k' is the number of parameters.
            For fitting purposes each initial guess is treated as a sperate trace to optimize.
    """
    num_traces = traces.shape[0]
    num_guesses = hyper_parameters.num_guesses

    parameters = parameter_ranges.to_parameters()

    # vmap over parameters
    log_likelihood_over_parameters = jax.vmap(
        lambda t, p, p_loc, p_scale: log_p_x_parameters(
            t, y, p, hyper_parameters, p_loc, p_scale
        ),
        in_axes=(None, 0, None, None),
    )

    # vmap over traces
    log_likelihoods = jax.vmap(log_likelihood_over_parameters, in_axes=(0, None, 0, 0))(
        traces,
        parameters,
        hyper_parameters.prior_locs,
        hyper_parameters.prior_scales,
    )

    # reshape parameters so they are "continuous" along each dimension
    parameters = parameters.reshape(parameter_ranges.num_values())

    # The following calls into non-JAX code and should therefore avoid vmap (or
    # any other transformation like jit or grad). That's why we use a for loop
    # to loop over traces instead of a vmap.
    guesses = []
    for i in range(num_traces):
        # reshape likelihoods to line up with parameters (so they are
        # "continuous" along each dimension)
        trace_log_likelihoods = log_likelihoods[i].reshape(
            parameter_ranges.num_values()
        )

        # find locations where parameters maximize log likelihoods
        min_index = find_maximum(trace_log_likelihoods)

        guesses.append(parameters[min_index])

    # all guesses are stored in 'guesses', the following stacks them together
    # as if we vmap'ed over traces:

    guesses = Parameters(
        jnp.stack([guesses[i].r_e for i in range(num_traces)]),
        jnp.stack([guesses[i].r_bg for i in range(num_traces)]),
        jnp.stack([guesses[i].mu_ro for i in range(num_traces)]),
        jnp.stack([guesses[i].sigma_ro for i in range(num_traces)]),
        jnp.stack([guesses[i].gain for i in range(num_traces)]),
        jnp.stack([guesses[i]._p_on_logit for i in range(num_traces)]),
        jnp.stack([guesses[i]._p_off_logit for i in range(num_traces)]),
        probs_are_logits=True,
    )

    return guesses


def is_done(log_likelihoods_history, hyper_parameters):
    """
    Determine if the parameter optimization has plateaued.

    Args:

        log_likelihood_history (deque):

            a deque containing the log likelihoods from the 'q' most recent iterations, where
            'q' is defined by :func:`HyperParameters.is_done_window`.

        hyper_parameters (:class:`HyperParameters`):

            The hyper-parameters used for the maximum likelihood estimation.


    Returns:

        Bool:

            if the imporvement over the last X iterations is below the limit
            defined in hyperparameters
    """

    if len(log_likelihoods_history) < hyper_parameters.is_done_window:
        return False

    # measures average percent change over last few cycles

    log_likelihoods_history = jnp.array(log_likelihoods_history)

    mean_values = jnp.abs(jnp.mean(log_likelihoods_history))
    mean_delta = jnp.abs(jnp.mean(jnp.diff(log_likelihoods_history, axis=0)))

    relative_improve = mean_delta / mean_values

    done_improve = relative_improve < hyper_parameters.is_done_limit

    # Check if nan and return true if so
    is_nan = jnp.isnan(relative_improve)
    converged = jnp.logical_or(done_improve, is_nan)

    return jnp.all(converged)
