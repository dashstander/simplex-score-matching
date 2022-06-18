import jax.numpy as jnp
import jax

def _proj_step(x):
    x = jnp.where(x > 0., x, 0.)
    num_pos = jnp.sum(x > 0)
    lambda_i = (1 - jnp.sum(x)) / num_pos
    x += lambda_i
    return x

def _vector_simplex_proj(x):
    """Orthogonal projection of a k-dimensional vector onto the k-simplex
    The "vector" algorithm from "Two Fast Algorithms for Projecting a Point on the Canonical Simplex" by Malozemov and Tamasyan
    @param x:        jnp.array
    @return x_proj:  jnp.array where jnp.sum(x_proj) == 1. 
    """
    k = x.shape[0]
    lambda_i = (1. - jnp.sum(x)) / k
    x0 = x + lambda_i
    x_proj = jax.lax.while_loop(
        lambda c: jnp.any(c < 0.),
        _proj_step,
        x0
    )
    return x_proj

# The _entire_ reason I'm using jax for this
vector_simplex_proj = jax.vmap(_vector_simplex_proj)
