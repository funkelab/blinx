from scipy.stats import entropy
import jax.numpy as jnp


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
    # find min likelihood and use it as value to replace bad values with
    sub_value = jnp.min(likelihoods[jnp.isfinite(likelihoods)])

    # remove nan likelihoods
    proc_likelihoods = likelihoods.at[jnp.isnan(likelihoods)].set(sub_value)

    # comapre differences in distributions
    dist_diffs = compare_dists(traces, parameters, hyper_parameters)
    likes_to_remove = dist_diffs > hyper_parameters.distribution_threshold

    proc_likelihoods = proc_likelihoods.at[likes_to_remove].set(sub_value)

    # find new most likely y values
    most_likely_ys = jnp.argmin(proc_likelihoods, axis=0) + \
        hyper_parameters.y_low

    return most_likely_ys, proc_likelihoods


def compare_dists(
        traces,
        parameters,
        hyper_parameters):
    # find the KL divergence between the measured viterbi distribution and the
    # theoretical distribution

    viterbi_traces, viterbi_dist = viterbi(
        traces,
        parameters,
        hyper_parameters.y_low)

    steady_state_dist = steady_state(parameters, hyper_parameters.y_low)

    kl = entropy(viterbi_dist, steady_state_dist, axis=2)

    return kl


"""
def steady_state(parameters, y_low):
    # theoretical steady state distribution of states for given parameters
    # calc using forward model of HMM

    ss_dists = np.zeros((
        parameters.shape[0],
        parameters.shape[1],
        parameters.shape[0]+y_low))
    for i, y in enumerate(range(y_low, y_low+parameters.shape[0])):
        for t in range(parameters.shape[1]):
            trans_matrix = transition_matrix.create_transition_matrix(
                y=y,
                p_on=parameters[i, t, PARAM_P_ON],
                p_off=parameters[i, t, PARAM_P_OFF],)

            ss_dists[i, t, :y+1] = transition_matrix.p_initial(y, trans_matrix)

    return ss_dists


def viterbi_single_trace(trace, params, y, bins):
    f_model = FluorescenceModel(
        mu_i=params[PARAM_MU],
        mu_b=params[PARAM_MU_BG],
        sigma=params[PARAM_SIGMA])
    t_model = TraceModel(
        f_model,
        p_on=params[PARAM_P_ON],
        p_off=params[PARAM_P_OFF])

    vit_trace = t_model.viterbi_alg(y, trace)
    vit_dist = jnp.histogram(
        vit_trace,
        density=True,
        bins=bins)

    return vit_trace, vit_dist


def viterbi(traces, parameters, y_low):

    ys = jnp.asarray(range(y_low, y_low+parameters.shape[0]))
    bins = jnp.asarray(range(parameters.shape[0] + 1 + y_low))

    vit_traces = []
    vit_dists = []
    for i in range(parameters.shape[0]):
        params = parameters[i, :, :]
        y = ys[i]

        a, b = vmap(viterbi_single_trace, in_axes=(0, 0, None, None))(
            traces, params, y, bins)
        vit_traces.append(a)
        vit_dists.append(b[0])

    return jnp.asarray(vit_traces), jnp.asarray(vit_dists)
"""
