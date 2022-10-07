import numpy as np
import scipy.stats as stats
from promap.fluorescence_model import FluorescenceModel
from promap.transition_matrix import TransitionMatrix
from jax import lax
import jax.numpy as jnp
from scipy.special import comb
import jax
from jax import random



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


    def set_params(self, p_on, p_off):

        self.p_on = jnp.float32(p_on)
        self.p_off = jnp.float32(p_off)

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

    def generate_trace(self, y, seed, distribution='lognormal'):
        # TODO: look into p_initial and breaking at high y values
        # think it is breaking because sum(p_initial > 1)
        # had same problem with _markov_trace
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
        TransMatrix = TransitionMatrix()
        transition_m = TransMatrix.create_transition_matrix(y, 
                                                                  self.p_on, 
                                                                  self.p_off)
        
        
        #transition_m = self.create_transition_matrix(y, self.p_on, self.p_off)
        
        # sum(rows) must always be = 1, rounding errors sometmes occur with 
        # small numbers, -> force sum(rows) <= 1
        rounding_error = jnp.clip(jnp.sum(transition_m, axis=1) - 1, a_min=0)
        max_locs = jnp.argmax(transition_m, axis=1)
        row_indicies = jnp.arange(0,y+1)
        transition_m = transition_m.at[row_indicies,max_locs].\
            set(transition_m[row_indicies,max_locs] - 2* rounding_error)
        
        p_initial = transition_m[0,:]
        # generate a list of states
        p_initial = jnp.log(transition_m[0,:])
        key = random.PRNGKey(seed)
        key, subkey = random.split(key)
        initial_state = jnp.expand_dims(random.categorical(subkey, p_initial), axis=0)

        # # generate a list of states
        
       
        #subkeys = jnp.expand_dims(random.split(key, num=self.num_frames), axis=2)
        subkeys = random.split(key, num=self.num_frames)
        
        scan2 = lambda state, key: self._scan(state, key, transition_m)
        
        a, states = jax.lax.scan(scan2, init=initial_state, xs=subkeys)
        
        if distribution == 'lognormal':
            sample_distribution = self.fluorescence_model.sample_x_z_lognorm_jax
        if distribution == 'poisson':
            sample_distribution = self.fluorescence_model.sample_x_z_poisson_jax
            
        key = random.PRNGKey(seed)
        subkey = random.split(key)
        x_trace = sample_distribution(jnp.asarray(states), subkey[0], shape=states.shape)

        return x_trace[:,0], states
    
    def _update_state(self, subkey, p_tr):
        state = random.categorical(subkey, p_tr)
        return state

    def _scan(self, old_state, key, transition_m):
        #key, subkey = random.split(key)
        p_tr = jnp.log(transition_m[old_state,:])
        new_state =self._update_state(key, p_tr)
        
        return new_state, new_state
   
    
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

        return -1*(jnp.sum(jnp.log(result)))

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
    
    
    