import numpy as np
from promap.fluorescence_model import FluorescenceModel
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
    '''

    def __init__(self, emission_params, p_on=None, p_off=None):

        # currently working with p_on/off, might need to switch to k_on/off
        self.p_on = p_on
        self.p_off = p_off
        self.fluorescence_model = FluorescenceModel(emission_params)

    def generate_trace(self, y, seed, num_frames, distribution='lognormal'):
        ''' generate a synthetic intensity trace

        Args:
            y (int):
                - The maximum number of elements that can be on

            seed (int):
                - random seed for the jax psudo rendom number generator

            distribution (string):
                - either 'lognormal' or 'poisson'
                - choice of distribution to sample intensities from

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

        # quick estiamte of p_initial values, WRONG!! (but works well enough)
        p_initial = jnp.log(transition_matrix.p_initial(y, transition_m))
        # jax.random.catigorical takes log probs

        # generate a list of states, use scan b/c state t depends on state t-1
        key = random.PRNGKey(seed)
        key, subkey = random.split(key)
        initial_state = jnp.expand_dims(random.categorical(subkey, p_initial),
                                        axis=0)

        scan2 = lambda state, key: self._scan_generate(state, key, transition_m)

        # add 100 frames, then remove first 100 to allow system to
        # come to equillibrium
        subkeys = random.split(key, num=num_frames+100)
        a, states = jax.lax.scan(scan2, init=initial_state, xs=subkeys)

        if distribution == 'lognormal':
            sample_distribution = self.fluorescence_model.sample_x_z_lognorm_jax
        if distribution == 'poisson':
            sample_distribution = self.fluorescence_model.sample_x_z_poisson_jax

        key = random.PRNGKey(seed)
        subkey = random.split(key)
        x_trace = sample_distribution(jnp.asarray(states), subkey[0],
                                      shape=states.shape)

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

    def _scan_generate(self, old_state, key, transition_m):
        p_tr = jnp.log(transition_m[old_state, :])
        new_state = random.categorical(key, p_tr)

        return new_state, new_state

    def _check_parameters(self):

        if self.p_on is None:
            raise RuntimeError("Parameters need to be set or fitted first.")

    def _scale_viterbi(self, x_trace, y, T, trans_m, p_init):
        "initialize"
        delta = np.zeros((y+1, T))
        sci = np.zeros((y+1, T))
        scale = np.zeros((T))
        ''' initial values '''
        for s in range(y+1):
            delta[s, 0] = p_init[s] * \
                self.fluorescence_model.p_x_i_given_z_i(x_trace[0], s)
        sci[:, 0] = 0

        ''' Propagation'''
        for t in range(1, T):
            for s in range(y+1):
                state_probs, ml_state = self._viterbi_mu(y, t, delta,
                                                         trans_m, s)
                delta[s, t] = state_probs * \
                    self.fluorescence_model.p_x_i_given_z_i(x_trace[t], s)
                sci[s, t] = ml_state
            scale[t] = 1 / np.sum(delta[:, t])
            delta[:, t] = delta[:, t] * scale[t]

        ''' build to optimal model trajectory output'''
        x = np.zeros((T))
        x[-1] = np.argmax(delta[:, T-1])
        for i in reversed(range(1, T)):
            x[i-1] = sci[int(x[i]), i]

        return x, delta, sci

    def _viterbi_mu(self, y, t, delta, trans_m, s):
        temp = np.zeros((y+1))
        for i in range(y+1):
            temp[i] = delta[i, t-1] * trans_m[i, s]
        return np.max(temp), np.argmax(temp)

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
    