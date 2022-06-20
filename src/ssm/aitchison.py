import chex
from functools import cache
import jax
import jax.numpy as jnp


def add(a: chex.Array, b: chex.Array) -> chex.Array:
    """ Vector addition on the simplex with the Aitchison geometry
    """
    x = jnp.multiply(a, b)
    return x / jnp.sum(x)


def mul(a, alpha) -> chex.Array:
    """ Scalar multiplication on the simplex with the Aitchison geometry
    """
    x =  jnp.pow(a, alpha)
    return x / jnp.sum(x)


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


@cache
def _vector(dim: int, i: int):
    x = jnp.zeros((dim,))
    x = x.at[0:i].set(1/i).at[i].set(-1)
    x *= jnp.sqrt(i / (i + 1))
    return x


@cache
def ortho_basis_rn(dim: int):
    vectors = [
        _vector(dim, i)[None] for i in range(1, dim)
    ]
    return jnp.concatenate(vectors)

@cache
def ortho_basis_simn(dim: int):
    return jax.vmap(clr_inv)(ortho_basis_rn(dim))
    
    
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
    
    
def riemann_inv_metric(x):
    """ The inverse metric tensor g^(-1) of the Riemannian metric on the simplex at point `x`. Also the Jacobian of the softmax at point `x`
    """
    d = x.shape[-1]
    
    def same_ind(i, j):
        return x[i] * (1. - x[i])

    def diff_ind(i, j):
        return -1. * x[i] * x[j]
    
    def _g(i):
        ks = jnp.arange(d)
        return jax.lax.map(
            lambda j: jax.lax.cond(i == j, same_ind, diff_ind, i, j),
            ks
        )
    return jax.lax.map(_g, jnp.arange(d))
    

