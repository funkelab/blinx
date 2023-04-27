import jax
import jax.numpy as jnp


def find_minima_nd(matrix, num_minima):
    '''
    Find local minima of an N dimensional matrix
    - finds nearest neighbors of each point
    - compares neighbors to determine if point is a minimum

    Returns:
        a N x num_minima array containing the coordinates of the found
        local minima
    '''

    # FIXME: has to be a way to avoid hard coding this
    # but not the worst because input params will always be 5d
    shape = matrix.shape
    dim_1 = jnp.arange(shape[0])
    dim_2 = jnp.arange(shape[1])
    dim_3 = jnp.arange(shape[2])
    dim_4 = jnp.arange(shape[3])
    dim_5 = jnp.arange(shape[4])

    num_elements = jnp.product(jnp.array(matrix.shape))

    b = jnp.asarray(jnp.meshgrid(
        dim_1, dim_2, dim_3, dim_4, dim_5
        )).reshape((matrix.ndim, num_elements)).T

    indices = jnp.arange(num_elements)
    d = jax.vmap(is_minimum_point, in_axes=(0, None, None))(indices, b, matrix)

    e = matrix.reshape(jnp.product(matrix.shape))

    return b[jnp.where(d == e, size=num_minima)]


def is_minimum_point(index, b, a):
    # Given a coordinate, finds the nearest neighbors and determines if given
    # coordinate is a local minimum
    centered = b - b[index]
    dist = jnp.linalg.norm(centered, axis=1)
    c = jnp.where(dist <= 1, size=len(a.shape*2)+1)
    tile = b[c]
    tile_values = a[tile[:, 0], tile[:, 1]]
    all_same = jnp.all(tile_values == tile_values[0])
    result = jax.lax.cond(all_same, lambda: 0., lambda: jnp.min(tile_values))
    return result
