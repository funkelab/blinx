import numpy as np
import scipy.stats as stats
from promap.fluorescence_model import FluorescenceModel
from jax import lax
import jax.numpy as jnp


class TraceModel:
    '''
    - Models an intensity trace as a hidden Markov model

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

    def __init__(self, emission_params, step_time, num_frames):

        # currently working with p_on/off, might need to switch to k_on/off
        self.p_on = None
        self.p_off = None
        self.step_time = step_time
        self.num_frames = num_frames
        self.fluorescence_model = FluorescenceModel(emission_params)

    def p_trace_given_y(self, trace, y):

        self._check_parameters()

        p_initial, transition_m = self._markov_trace(y)
        log_fwrd_prob = self._forward_alg(trace, y, transition_m, p_initial)

        return log_fwrd_prob

    def p_trace_given_probs(self, trace, y):
        self._check_parameters()

        p_initial, transition_m = self._markov_trace(y)
        probs = self.fluorescence_model.vmap_p_x_given_z(trace, y)
        log_fwrd_prob = self._forward_alg_jax(probs, transition_m, p_initial)

        return log_fwrd_prob

    def set_params(self, p_on, p_off):

        self.p_on = p_on
        self.p_off = p_off

    def fit_params(self, traces, y, method='line_search', **kwargs):
        '''
        Fit all the parameters needed for the trace model
            p_on, p_off
        '''

        if method == 'line_search':
            self._line_search_params(traces, y, **kwargs)
        elif method == 'viterbi':
            self._viterbi_fit_params(traces, y, **kwargs)
        return

    def generate_trace(self, y):

        self._check_parameters()

        p_initial, transition_m = self._markov_trace(y)
        # generate list of states
        initial_state = list(stats.multinomial.rvs(1, p_initial)).index(1)
        states = [initial_state]
        for i in range(self.num_frames-1):
            p_tr = transition_m[states[-1], :]
            new_state = list(stats.multinomial.rvs(1, p_tr)).index(1)
            states.append(new_state)

        # generate observations from list of states
        x_trace = np.ones((len(states)))
        for i in range(len(states)):
            x_trace[i] = self.fluorescence_model.sample_x_i_given_z_i(states[i])

        return x_trace, states

    def estimate_y(self, trace, guess, search_width):

        self._check_parameters()

        log_probs = np.zeros((search_width*2+1))
        low_bound = 0 if guess - search_width < 0 else guess - search_width
        for i, y in enumerate(range(low_bound, guess+search_width+1)):
            log_probs[i] = self.p_trace_given_y(trace, y)

        return log_probs

    def _check_parameters(self):

        if self.p_on is None:
            raise RuntimeError("Parameters need to be set or fitted first.")

    def _markov_trace(self, y):
        c_state = np.ones(y+1) / (y+1)
        transition_m = self._create_transition_m(y)
        prob_trace = np.zeros((y+1, 10))
        for i in range(10):
            c_state = c_state @ transition_m
            prob_trace[:, i] = c_state
        p_initial = prob_trace[:, -1]

        # correct minor rounding errors
        if np.sum(p_initial) > 1.0:
            pos = np.argmax(p_initial[:])
            p_initial[pos] = p_initial[pos] - (np.sum(p_initial)-1)

        # check and fix minor rounding errors in transition matrix
        # -- maybe have np round down in _create_transition_m
        for i in range(transition_m.shape[1]):
            prob_total = np.sum(transition_m[i, :])
            if prob_total > 1.0:
                pos = np.argmax(transition_m[i, :])
                transition_m[i, pos] = transition_m[i, pos] - (prob_total-1)

        return p_initial, transition_m

    def _create_transition_m(self, y):

        size = y+1  # possible states range from 0 - y inclusive
        transition_m = np.zeros((size, size))

        for i in range(size):
            for j in range(size):
                p = 0
                for z in range(i+1):
                    p += stats.binom.pmf(z, i, self.p_off) * \
                        stats.binom.pmf(j-i+z, y-i, self.p_on)
                transition_m[i, j] = p

        return transition_m

    def _forward_alg(self, x_trace, y, transition_m, p_init):

        forward = np.zeros((y+1, len(x_trace)))
        scale_factors = np.zeros((len(x_trace)))

        # Initialize
        for s in range(y+1):
            forward[s, 0] = p_init[s] * \
                self.fluorescence_model.p_x_i_given_z_i(x_trace[0], s)
        scale_factors[0] = 1 / np.sum(forward[:, 0])
        forward[:, 0] = forward[:, 0] * scale_factors[0]

        # Propagate
        for t in range(1, len(x_trace)):
            for state in range(y+1):
                forward[state, t] = self._alpha(forward, x_trace, transition_m,
                                                y, state, t)

            scale_factors[t] = 1 / np.sum(forward[:, t])
            forward[:, t] = forward[:, t] * scale_factors[t]

        # Total likelyhood
        log_fwrd_prob = -1*(np.sum(np.log(scale_factors)))

        return log_fwrd_prob

    def _alpha(self, forward, x_trace, transition_m, y, s, t):
        # recursive element of the forward algorithm
        p = 0
        for i in range(y+1):
            p += forward[i, (t-1)] * transition_m[i, s] * \
                self.fluorescence_model.p_x_i_given_z_i(x_trace[t], s)

        return p
    
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
                state_probs, ml_state = self._viterbi_mu(y, t, delta, trans_m, s)
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


    def _forward_alg_jax(self, probs, transition_m, p_init):
        initial_values = p_init[:] * probs[:, 0]
        scale_factor_initial = 1 / jnp.sum(initial_values)
        initial_values = initial_values * scale_factor_initial
        p_transition = transition_m

        scan_f_2 = lambda p_accumulate, p_emission: self._scan_f(p_accumulate,
                                                                 p_emission, 
                                                                 p_transition)

        final, result = lax.scan(scan_f_2, initial_values, probs.T)

        return -1*(np.sum(np.log(result)))

    def _scan_f(self, p_accumulate, p_emission, p_transition):
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
