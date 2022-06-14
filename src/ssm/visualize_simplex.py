import chex
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

"""Functions to visualize stuff happening on the 3-simplex
"""

# The idea here is that the 3-simplex is an equilateral triangle with sidelengths
# of sqrt(2). Each point on the triangle ((1, 0, ), (0, 1, 0), (0, 0, 1)) represents
# the categorical distribution on three elements where you sample element {1,2,3} with probability 1.
# Defining a transformation from elements of the 3-simplex to the triangle with points at


_sq32 = jnp.sqrt(3/2)
_sq22 = jnp.sqrt(2)/2

def rotate_simplex_to_xy(x):
    rot = jnp.array([[-_sq22, _sq22, 0.], [0., 0., _sq32], [0., 0., 0.]])
    return jnp.matmul(rot, x)
