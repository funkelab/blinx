import jax.numpy as jnp
import numpy as np
import scipy.signal


def find_maximum(matrix):
    temp_matrix = np.array(matrix)

    # argmax will return index of any nans
    # so replace nans with -inf

    mask = np.isnan(temp_matrix)

    temp_matrix[mask] = -np.inf

    index = np.argmax(temp_matrix)

    index = np.unravel_index(index, temp_matrix.shape)

    return tuple(jnp.expand_dims(jnp.array(i), axis=-1) for i in index)
