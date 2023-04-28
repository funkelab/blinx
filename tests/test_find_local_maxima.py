from promap.utils import find_local_maxima
import jax.numpy as jnp
import numpy as np


def test_single_maximum():
    # one maximum, somewhere within the matrix

    a = np.zeros((10, 10), dtype=np.float32)
    a[3, 4] = 2.0
    a = jnp.array(a)

    maximum_indices = find_local_maxima(a, num_maxima=1)

    assert len(maximum_indices[0]) == 1
    np.testing.assert_equal(np.asarray(maximum_indices), np.asarray(([3], [4])))

    # one maximum, at the left boundary

    a = np.zeros((10, 10), dtype=np.float32)
    a[0, 4] = 2.0
    a = jnp.array(a)

    maximum_indices = find_local_maxima(a, num_maxima=1)

    assert len(maximum_indices[0]) == 1
    np.testing.assert_equal(np.asarray(maximum_indices), np.asarray(([0], [4])))

    # one maximum, at the right boundary

    a = np.zeros((10, 10), dtype=np.float32)
    a[9, 4] = 2.0
    a = jnp.array(a)

    maximum_indices = find_local_maxima(a, num_maxima=1)

    assert len(maximum_indices[0]) == 1
    np.testing.assert_equal(np.asarray(maximum_indices), np.asarray(([9], [4])))

    # one maximum, at the corner

    a = np.zeros((10, 10), dtype=np.float32)
    a[9, 9] = 2.0
    a = jnp.array(a)

    maximum_indices = find_local_maxima(a, num_maxima=1)

    assert len(maximum_indices[0]) == 1
    np.testing.assert_equal(np.asarray(maximum_indices), np.asarray(([9], [9])))


def test_multiple_maxima():
    a = np.zeros((10, 10), dtype=np.float32)
    a[3, 4] = 1.5
    a[9, 9] = 2.0
    a = jnp.array(a)

    maximum_indices = find_local_maxima(a)

    assert len(maximum_indices[0]) == 2
    np.testing.assert_equal(np.asarray(maximum_indices), np.asarray(([3, 9], [4, 9])))


def test_no_maximum():
    a = np.zeros((10, 10), dtype=np.float32)
    a = jnp.array(a)

    maximum_indices = find_local_maxima(a)

    assert len(maximum_indices[0]) == 0


def test_multi_dimensional():
    a = np.zeros((10, 9, 8, 7, 6, 5, 4, 3, 2), dtype=np.float32)
    a[5, 5, 4, 4, 3, 3, 2, 2, 1] = 2.0
    a = jnp.array(a)

    maximum_indices = find_local_maxima(a)

    assert len(maximum_indices[0]) == 1
    np.testing.assert_equal(
        np.asarray(maximum_indices),
        np.asarray(([5], [5], [4], [4], [3], [3], [2], [2], [1])),
    )
