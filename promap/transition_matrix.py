import jax
import jax.numpy as jnp
from scipy.special import comb



class TransitionMatrix:
    
    def __init__(self):
        
        return
    
    
    def create_transition_matrix(self,
            y,
            p_on,
            p_off,
            comb_matrix=None,
            comb_matrix_slanted=None):
        '''Create a transition matrix for the number of active elements, given that
        elements can randomly turn on and off.

        Args:

            y (int):
                The maximum number of elements that can be on.

            p_on (float):
                The probability for a single element to turn on (if off).

            p_off (float):
                The probability for a single element to turn off (if on).

            comb_matrix (array, optional):
            comb_matrix_slanted (array, optional):
                Precomputed combinatorial matrices, containing the counts of
                possible flips of ``i`` elements out of ``j`` elements. If not
                given, those matrices will be created inside this function.
                However, since the values in those matrices do not depend on
                ``p_on`` and ``p_off`, it will be more efficient to precompute
                those matrices and pass them here::

                    comb_matrix = create_comb_matrix(y)
                    comb_matrix_slanted = create_comb_matrix(y, slanted=True)

        Returns:

            A matrix of transition probabilities of shape ``(y + 1, y + 1)``, with
            element ``i, j`` being the probability that the number of active
            elements changes from ``i`` to ``j``.
        '''

        p_on = jnp.float32(p_on)
        p_off = jnp.float32(p_off)
        if comb_matrix is None:
             comb_matrix = self._create_comb_matrix(y)
        if comb_matrix_slanted is None:
             comb_matrix_slanted = self._create_comb_matrix(y, slanted=True)

        max_y = comb_matrix.shape[0] - 1

        prob_matrix_on = self._create_prob_matrix(y, jnp.float32(p_on), slanted=True)
        prob_matrix_off = self._create_prob_matrix(y, jnp.float32(p_off))

        t_on_matrix = comb_matrix_slanted * prob_matrix_on
        t_off_matrix = comb_matrix * prob_matrix_off

        def correlate(t_on_matrix, t_off_matrix):
            return jax.vmap(
                lambda a, b: jnp.correlate(a, b, mode='valid')
            )(t_on_matrix[::-1], t_off_matrix)

        return correlate(t_on_matrix[:y + 1, max_y - y:], t_off_matrix[:y + 1])
    
    def _create_comb_matrix(self, y, slanted=False):
        '''Creates a matrix of n-choose-k values.

        Args:

            y (int):
                The maximum number of elements. The returned matrix will have shape
                ``(y + 1, y + 1)``.

            slanted (bool):
                If given, the returned matrix will be "slanted" to the right, i.e.,
                the second last row will be shifted by 1, the third last one by 2,
                and so on. The shape of the returned matrix will then be ``(y + 1,
                2 * y + 1)``. The slanted form is used to facilitate computation of
                a square transition matrix.

        Returns:

            A matrix of n-choose-k values of shape ``(y + 1, y + 1)``, such that
            the element at position ``i, j`` is the number of ways to select ``j``
            elements from ``i`` elements.
        '''

        end_i = y + 1
        end_j = y + 1 if not slanted else 2 * y + 1

        if slanted:
            return jnp.array([
                [comb(i, j - (y - i)) for j in range(end_j)]
                for i in range(end_i)
            ])
        else:
            return jnp.array([
                [comb(i, j) for j in range(end_j)]
                for i in range(end_i)
            ])
        
    def _create_prob_matrix(self, y, p, slanted=False):
        '''Creates a matrix of probabilities for flipping ``i`` out of ``j``
        elements, given that the probability for a single flip is ``p``.

        Args:

            y (int):
                The maximum number of elements. The returned matrix will have shape
                ``(y + 1, y + 1)``.

            p (float):
                The probability of a single flip.

            slanted (bool):
                If given, the returned matrix will be "slanted" to the right, i.e.,
                the second last row will be shifted by 1, the third last one by 2,
                and so on. The shape of the returned matrix will then be ``(y + 1,
                2 * y + 1)``. The slanted form is used to facilitate computation of
                a square transition matrix.

        Returns:

            A matrix of probabilities of shape ``(y + 1, y + 1)``, such that the
            element at position ``i, j`` is the probability to flip ``j`` elements
            out of ``i`` elements, if the probability for a single flip is ``p``.
        '''

        i_indices = jnp.arange(0, y + 1)
        j_indices = jnp.arange(0, 2 * y + 1) if slanted else jnp.arange(0, y + 1)

        def prob_i_j(i, j):
            # i are on, j flip
            # -> i - j stay
            # -> j flip
            return p**j * (1.0 - p)**(i - j)

        def prob_i(i):
            if slanted:
                def prob_i_fun(j):
                    return prob_i_j(i, j - (y - i))
            else:
                def prob_i_fun(j):
                    return prob_i_j(i, j)
            return jax.vmap(prob_i_fun)(j_indices)

        return jax.vmap(prob_i)(i_indices)