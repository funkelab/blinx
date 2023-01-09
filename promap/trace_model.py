import numpy as np
from promap import transition_matrix
from jax import lax
import jax.numpy as jnp
import jax
from jax import random


class TraceModel:
    '''
    TODO: replace with a better docstring

    Args:

        p_on:
            the probability of a single strand of imager DNA binding to
            a single docker during time (step_time)

        p_off:
            the probability of a bound imager and docker strand
            dissociating during time (step_time)

        step_time (seconds):
            length of a single measurement, equivalent to the camera exposure
            time or 1 / camera frame rate

        num_frames:
            the number of frames in the trace, multiply by step time to get
            length of trace

        distribution (string):
            - either 'lognormal' or 'poisson'
            - choice of distribution to sample intensities from
    '''

    def __init__(self, fluorescence_model, p_on=None, p_off=None):

        # currently working with p_on/off, might need to switch to k_on/off
        self.fluorescence_model = fluorescence_model
        self.p_on = p_on
        self.p_off = p_off

    def generate_trace(self, y, seed, num_frames):
        ''' generate a synthetic intensity trace

        Args:
            y (int):
                - the total number of fluorescent emitters

            seed (int):
                - random seed for the jax psudo rendom number generator

            num_frames (int):
                - the number of observations to simulate

        Returns:
            x_trace (array):
                - an ordered array of length num_frames containing intensity
                    values for each frame

            states (array):
                - array the same shape as x_trace, showing the hiddens state
                    "z" for each frame
                '''

        transition_m = transition_matrix.create_transition_matrix(y, self.p_on,
                                                                  self.p_off)

        # sum(rows) must always be = 1, rounding errors sometimes occur with
        # small numbers, -> force sum(rows) <= 1
        rounding_error = jnp.clip(jnp.sum(transition_m, axis=1) - 1, a_min=0)
        max_locs = jnp.argmax(transition_m, axis=1)
        row_indicies = jnp.arange(0, y+1)
        transition_m = transition_m.at[row_indicies, max_locs].\
            set(transition_m[row_indicies, max_locs] - 2 * rounding_error)

        p_initial = jnp.log(transition_matrix.p_initial(y, transition_m))
        # jax.random.catigorical takes log probs

        # generate a list of states, use scan b/c state t depends on state t-1
        key = random.PRNGKey(seed)
        key, subkey = random.split(key)
        initial_state = jnp.expand_dims(random.categorical(subkey, p_initial),
                                        axis=0)

        scan2 = lambda state, key: self._scan_generate(state, key,
            transition_m)

        # add 100 frames, then remove first 100 to allow system to
        # come to equillibrium
        subkeys = random.split(key, num=num_frames+100)
        a, states = jax.lax.scan(scan2, init=initial_state, xs=subkeys)

        key = random.PRNGKey(seed)
        subkey = random.split(key)
        x_trace = self.fluorescence_model.sample_x_z(
            jnp.asarray(states),
            subkey[0])

        return x_trace[100:, 0], states[100:]

    def get_likelihood(self, probs, transition_m, p_init):
        '''
        TODO: add a docstring
        '''
        initial_values = p_init[:] * probs[:, 0]
        scale_factor_initial = 1 / jnp.sum(initial_values)
        initial_values = initial_values * scale_factor_initial
        p_transition = transition_m

        scan_f_2 = lambda p_accumulate, p_emission: self._scan_likelihood(
                                                                 p_accumulate,
                                                                 p_emission,
                                                                 p_transition)

        final, result = lax.scan(scan_f_2, initial_values, probs.T)

        return -1*(jnp.sum(jnp.log(result)))

    def viterbi_alg(self, y, trace):
        '''
        Find the most likely state for each frame of the trace

        Args:
            - y (int):
                - the assumed total number of fluorescent emitters

            - trace (array):
                - the observation sequence

        Returns:
            - s_opt (array):
                - the optimal or most likely sequence of states
        '''
        probs = self.fluorescence_model.p_x_given_zs(trace, y)
        trans_m = transition_matrix.create_transition_matrix(y,
             self.p_on, self.p_off)
        p_init = transition_matrix.p_initial(y, trans_m)

        num_states = trans_m.shape[0]
        num_obs = len(trace)  # length of observed sequence
        tiny = np.finfo(0.).tiny
        trans_m_log = np.log(trans_m + tiny)
        p_init_log = np.log(p_init + tiny)
        probs_log = np.log(probs + tiny)

        delta_log = np.zeros((num_states, num_obs))
        E = np.zeros((num_states, num_obs-1)).astype(np.int32)
        delta_log[:, 0] = p_init_log + probs_log[:, 0]

        for n in range(1, num_obs):
            for i in range(num_states):
                temp_sum = trans_m_log[:, i] + delta_log[:, n-1]
                delta_log[i, n] = np.max(temp_sum) + probs_log[i, n]
                E[i, n-1] = np.argmax(temp_sum)
        s_opt = np.zeros(num_obs).astype(np.int32)
        s_opt[-1] = np.argmax(delta_log[:, -1])
        for n in range(num_obs-2, -1, -1):
            s_opt[n] = E[int(s_opt[n+1]), n]

        return s_opt

    def _scan_generate(self, old_state, key, transition_m):
        p_tr = jnp.log(transition_m[old_state, :])
        new_state = random.categorical(key, p_tr)

        return new_state, new_state

    def _check_parameters(self):

        if self.p_on is None:
            raise RuntimeError("Parameters need to be set or fitted first.")

    def _scan_likelihood(self, p_accumulate, p_emission, p_transition):
        '''
        p_accu:
            - accumulated probability from t=0 to t-1
            - vector shape (1 x Y)

        p_emission:
            - probability of observing X given state z
            - precalculated and stored in a array shape (t x y)

        p_transtion:
            - probability of transitioning from state z(t-1) to state z(t)
            - precalculated and stored in an array shape (y x y)

        '''
        temp = p_emission * jnp.matmul(p_accumulate, p_transition)
        scale_factor = 1 / jnp.sum(temp)
        prob_time_t = temp * scale_factor

        return prob_time_t, scale_factor
