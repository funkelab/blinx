import jax.numpy as jnp
import numpy as np
import scipy.signal


def find_local_maxima(matrix, num_maxima=None):
    # convert to numpy array
    matrix = np.asarray(matrix)

    if num_maxima is None:
        num_maxima = matrix.size

    # pad matrix with -inf
    padded = np.ones(tuple(s + 2 for s in matrix.shape), dtype=matrix.dtype)
    padded *= -np.inf
    slices = tuple(slice(1, s + 1) for s in matrix.shape)
    padded[slices] = matrix

    padded_indices = scipy.signal.argrelmax(np.asarray(padded), mode="wrap")
    # indices into original matrix without padding
    indices = tuple(i - 1 for i in padded_indices)

    # set all non-maxima to -inf
    maxima = np.ones_like(matrix)
    maxima *= -np.inf
    maxima[indices] = matrix[indices]

    # get all maximum values, sorted
    values, indices = np.unique(maxima, return_index=True)

    assert values[0] == -np.inf

    # first index should point to -np.inf, drop it
    indices = indices[1:]

    # retain only last num_maxima values
    if len(indices) > num_maxima:
        indices = indices[-num_maxima:]

    # convert back to non-flattened indices
    indices = np.unravel_index(indices, matrix.shape)

    # convert to jax array
    indices = tuple(jnp.array(i) for i in indices)

    return indices
