import jax
import jax.numpy as jnp

from .constants import eps


def get_steady_state(transition_matrix):
    num_states = transition_matrix.shape[0]

    # initialize with a uniform distribution
    initial_state = jnp.ones(num_states) / num_states

    steady_state, _ = jax.lax.scan(
        lambda state, _: (jnp.matmul(state, transition_matrix), None),
        initial_state,
        xs=None,
        length=100,
    )

    return steady_state


def get_measurement_log_likelihood(
    p_measurement, p_initial, p_transition
):
    # for each timestep, we have:
    #
    # a "state"   a discrete variable (0, 1, ...)
    # a "pstate"  a categorical probability distribution over the state
    #
    # The following implements the forward algorithm, which only considers
    # "pstate"s.

    # p_measurement: (t, y + 1)

    def get_next_pstate(prev_pstate, p_measurement):
        next_pstate = p_measurement * jnp.matmul(prev_pstate, p_transition)

        normalization_factor = 1.0 / jnp.sum(next_pstate)
        next_pstate = next_pstate * normalization_factor

        return next_pstate, normalization_factor

    # t = 0

    initial_pstate = p_initial * p_measurement[0]
    normalization_factor = 1.0 / jnp.sum(initial_pstate)
    initial_pstate = initial_pstate * normalization_factor

    # t = 1, 2, ...

    final_pstate, normalization_factors = jax.lax.scan(
        get_next_pstate, initial_pstate, p_measurement[1:]
    )

    # The final likelihood is:
    #
    # likelihood = sum(final_pstate) / prod(normalization_factors)
    #
    # but by definition, jnp.sum(final_pstate) == 1, so we skip that.
    # We also compute the log likelihood to avoid an expensive exp:
    #
    # log_likelihood = log(sum(final_pstate) / prod(normalization_factors))
    #                = log(sum(final_pstate)) - log(prod(normalization_factors))
    #                = log(1.0) - sum(log(normalization_factors))
    #                = 0.0 - sum(log(normalization_factors))
    #                = - sum(log(normalization_factors))

    log_likelihood = -(
        jnp.sum(jnp.log(normalization_factors)) + jnp.log(normalization_factor)
    )

    return log_likelihood


def get_optimal_states(measurements, p_measurement, p_initial, p_transition):
    # for each timestep, we have:
    #
    # a "state"   a discrete variable (0, 1, ...)
    # a "pstate"  a categorical probability distribution over the state
    #
    # The following implements the Viterbi algorithm to find the optimal
    # sequence of states.

    # turn all probabilities into log probabilities and ensure that log(.) is
    # well behaved by clipping probabilities with eps
    log_p_measurement = jnp.log(jnp.clip(p_measurement, a_min=eps))
    log_p_initial = jnp.log(jnp.clip(p_initial, a_min=eps))
    # We use this opportunity to transpose the transition probability matrix,
    # to retain the semantics of the equivalent matrix multiplication we carry
    # out in the forward algorithm. In essence, matmul(pstate, p_transition)
    # is congruent to log(pstate) + log(p_transition).T. We transpose here to
    # avoid having to do it in the scan.
    log_p_transition = jnp.log(jnp.clip(p_transition, a_min=eps)).T

    # t = 0

    log_initial_pstate = log_p_initial + log_p_measurement[measurements[0]]

    # t = 1, 2, ...

    def get_next_log_pstate(prev_log_pstate, measurement):
        # prev_log_pstate           (n,)    MAP log probability of previous state
        #
        # log_p_transition          (n, n)  Transpose of log probability
        #                                   matrix, i.e., element [i,j] is the
        #                                   log probability of arriving at
        #                                   state i from state j
        #
        # joint_log_pstate          (n, n)  Joint log probability after
        #                                   transition. Element [i,j] is the
        #                                   log probability of having
        #                                   transitioned to state i after
        #                                   having been in state j.
        #
        # forward_log_pstate        (n,)    Row-wise max of joint_log_pstate.
        #                                   Element i is the highest log
        #                                   probability of having arrived in
        #                                   state i from any previous state j.
        #
        # best_prev_state_lookup    (n,)    Row-wise argmax of
        #                                   joint_log_pstate. Element i is the
        #                                   most likely previous state j to
        #                                   have arrived from.
        #
        # next_log_pstate           (n,)    MAP log probability of next state.

        joint_log_pstate = prev_log_pstate + log_p_transition
        forward_log_pstate = jnp.max(joint_log_pstate, axis=1)
        best_prev_state_lookup = jnp.argmax(joint_log_pstate, axis=1)
        next_log_pstate = forward_log_pstate + log_p_measurement[measurement]

        return next_log_pstate, best_prev_state_lookup

    final_log_pstate, best_prev_state_lookups = jax.lax.scan(
        get_next_log_pstate, log_initial_pstate, measurements[1:]
    )

    # final best state is the argmax of the final log probability

    final_best_state = jnp.argmax(final_log_pstate)

    # all other previous best states can now be traced backwards from there

    _, backward_trace = jax.lax.scan(
        lambda best_state, best_prev_state_lookup: (best_prev_state_lookup[best_state],)
        * 2,
        final_best_state,
        best_prev_state_lookups,
        reverse=True,
    )

    # append the final best state to the traced back states
    return jnp.concatenate([backward_trace, jnp.array([final_best_state])])
