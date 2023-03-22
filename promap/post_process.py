import numpy as np
from promap.constants import (
    PARAM_MU,
    PARAM_MU_BG,
    PARAM_SIGMA,
    PARAM_P_ON,
    PARAM_P_OFF)
from promap.fluorescence_model import FluorescenceModel
from promap.trace_model import TraceModel
from promap import transition_matrix
import jax.numpy as jnp
from scipy.stats import entropy
from jax import vmap


def post_process(
        traces,
        parameters,
        likelihoods,
        hyper_parameters):
    '''
    Big wrapper function that combines all post processing steps and returns
    the best y guess for each trace

    Inputs: traces, parameters, likelihoods, hyper_parameters

    '''
    # find max likelihood and use it as value to replace bad values with
    sub_value = jnp.max(likelihoods[jnp.isfinite(likelihoods)])

    # remove nan likelihoods
    proc_likelihoods = likelihoods.at[jnp.isnan(likelihoods)].set(sub_value)

    # comapre differences in distributions
    dist_diffs = _compare_dists(traces, parameters, hyper_parameters)
    likes_to_remove = dist_diffs > hyper_parameters.distribution_threshold

    proc_likelihoods = proc_likelihoods.at[likes_to_remove].set(sub_value)

    # find new most likely y values
    most_likely_ys = jnp.argmin(proc_likelihoods, axis=0) + \
        hyper_parameters.y_low

    return most_likely_ys, proc_likelihoods


def _compare_dists(
        traces,
        parameters,
        hyper_parameters):
    # find the KL divergence between the measured viterbi distribution and the
    # theoretical distribution

    viterbi_traces, viterbi_dist = _viterbi(
        traces,
        parameters,
        hyper_parameters.y_low)

    steady_state_dist = _steady_state(parameters, hyper_parameters.y_low)

    kl = entropy(viterbi_dist, steady_state_dist, axis=2)

    return kl


def _viterbi(traces, params, y_low):
    # Use viterbi algorithm to fit traces to given parameters
    # return viterbi trace and distribution of states

    viterbi_traces = np.zeros((
        params.shape[0],
        traces.shape[0],
        traces.shape[1]))
    vit_dists = np.zeros((
        params.shape[0],
        params.shape[1],
        params.shape[0] + y_low))
    for i, y in enumerate(range(y_low, y_low+params.shape[0])):
        for t in range(traces.shape[0]):
            f_model = FluorescenceModel(
                mu_i=params[i, t, PARAM_MU],
                mu_b=params[i, t, PARAM_MU_BG],
                sigma=params[i, t, PARAM_SIGMA])
            t_model = TraceModel(
                f_model,
                p_on=params[i, t, PARAM_P_ON],
                p_off=params[i, t, PARAM_P_OFF])

            vit_trace = t_model.viterbi_alg(y, traces[t, :])
            vit_dists[i, t, :], _ = np.histogram(
                vit_trace,
                density=True,
                bins=range(params.shape[0] + 1 + y_low))

            viterbi_traces[i, t, :] = vit_trace

    return viterbi_traces, vit_dists


def _steady_state(parameters, y_low):
    # theoretical steady state distribution of states for given parameters
    # calc using forward model of HMM

    ss_dists = np.zeros((
        parameters.shape[0],
        parameters.shape[1],
        parameters.shape[0]+y_low))
    print(f'parameters: {parameters.shape}')
    print(f'ss_dists: {ss_dists.shape}')
    for i, y in enumerate(range(y_low, y_low+parameters.shape[0])):
        for t in range(parameters.shape[1]):
            trans_matrix = transition_matrix.create_transition_matrix(
                y=y,
                p_on=parameters[i, t, PARAM_P_ON],
                p_off=parameters[i, t, PARAM_P_OFF],)

            ss_dists[i, t, :y+1] = transition_matrix.p_initial(y, trans_matrix)

    return ss_dists
