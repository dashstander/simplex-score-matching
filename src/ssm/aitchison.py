from typing import Callable
import chex
from functools import lru_cache
import jax
import jax.numpy as jnp
from typing import Callable, Tuple


def add(a: chex.Array, b: chex.Array) -> chex.Array:
    """ Vector addition on the simplex with the Aitchison geometry
    """
    x = jnp.multiply(a, b)
    return x / jnp.sum(x)


def mul(a, alpha) -> chex.Array:
    """ Scalar multiplication on the simplex with the Aitchison geometry
    """
    x =  jnp.power(a, alpha)
    return x / jnp.sum(x)


@jax.jit
def inverse(x):
  x = jax.lax.clamp(1e-14, x, 1e14)
  return jnp.reciprocal(x)


def distance(x, y):
    aitch_dist = clr(x) - clr(y)
    return jnp.dot(aitch_dist, aitch_dist)


def aitch_basis(dim: int):
    total = dim - 1. + jnp.e
    basis = jnp.ones((dim, dim))
    i = jnp.diag_indices(dim)
    return basis.at[i].set(jnp.e) / total


def clr_inv(x: chex.Array, axis=-1, initial=None) -> chex.Array:
    """ The inverse of the CLR transform. Just the softmax, but the code makes more sense with the aliasing
    """
    return jax.nn.softmax(x, axis=axis, initial=initial)


def clr(x: chex.Array) -> chex.Array:
    """ Centered log ration (clr) transform of a point on the simplex. Takes a point in the canonical basis to 
    """
    log_x = jnp.log(x)
    geom_mean = jnp.exp(jnp.mean(log_x))
    return jnp.log(x / geom_mean)


@jax.jit
def aitch_dot(a: chex.Array, b: chex.Array) -> chex.Scalar:
    """ Inner product between two elements of the simplex with the Aitchison geometry
    """
    return jnp.dot(clr(a), clr(b))


def ortho_basis_rn(dim):
    def j_less(i):
        return 1 / i
      
    def j_equal(i):
        return -1.
  

    def basis_fn(i, j):
        i = i + 1
        j = j + 1
        val = jax.lax.cond(
            j <= i,
            j_less,
            lambda index: jax.lax.cond(j == (i + 1), j_equal, lambda _: 0., i),
            i
        )
        return val * jnp.sqrt(i / (i + 1))
    
    return jnp.fromfunction(basis_fn, (dim - 1, dim))


def ortho_basis_simn(dim: int):
    return jax.vmap(clr_inv)(ortho_basis_rn(dim))


def make_isometric_transforms(dim: int) -> Tuple[Callable]:
    rn_basis = ortho_basis_rn(dim)

    def ilr(x):
        return jnp.matmul(rn_basis, clr(x))
    
    def ilr_inv(y):
        return clr_inv(jnp.matmul(jnp.transpose(rn_basis), y))
    
    return ilr, ilr_inv
    
    
def ilr(x):
    """ x in Sim^D, this function sends it to R^(D-1) according to an orthonormal basis
    """
    d = x.shape[-1]
    ortho = ortho_basis_rn(d)
    return jnp.matmul(ortho, clr(x))
    

def ilr_inv(y):
    d = y.shape[-1]
    basis = ortho_basis_rn(d + 1)
    return clr_inv(jnp.matmul(jnp.transpose(basis), y))
    
    
def simplex_metric_tensor_inv(x, v):
    _, g_inv = jax.jvp(
        jax.nn.softmax,
        (x, ),
        (v, )
    )
    return g_inv

