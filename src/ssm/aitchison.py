import chex
import itertools
import jax
import jax.numpy as jnp


def perturb(a: chex.Array, b: chex.Array) -> chex.Array:
    """ Vector addition on the simplex with the Aitchison geometry
    """
    x = jnp.multiply(a, b)
    return x / jnp.sum(x)


def pow(a, alpha) -> chex.Array:
    """ Scalar multiplication on the simplex with the Aitchison geometry
    """
    x =  a ** alpha
    return x / jnp.sum(x)


@jax.jit
def _index_pairs(dim: int)-> chex.Array:
    return jnp.array(list(itertools.combinations(range(dim), 2)))


def circulant(x: chex.Array) -> chex.Array:
    """ Takes a vector (i.e. shape "(k,)") and returns the circulant k x k matrix
    # TODO: Can I use this to cleverly calculate the Aitchison inner product?
    """
    chex.assert_rank()
    size = x.size
    circ = jnp.concatenate(
        [jnp.roll(x, i)[None] for i in range(size)]
    )
    return circ.T


@jax.jit
def aitch_dot(a: chex.Array, b: chex.Array) -> chex.Scalar:
    """ Inner product between two elements of the simplex with the Aitchison geometry
    """
    d = a.size[-1]
    indices =  jnp.array(_index_pairs(d))
    def pairwise(val, pair):
        i, j = pair[0], pair[1]
        sum1 = jnp.log(a[i])/jnp.log(a[j]) * jnp.log(b[i])/jnp.log(b[j])
        sum2 = jnp.log(a[j])/jnp.log(a[i]) * jnp.log(b[j])/jnp.log(b[i])
        return val + sum1 + sum2
    return jax.lax.scan(pairwise, 0., indices)



def aitch_basis(dim: int):
    total = dim - 1. + jnp.e
    basis = jnp.ones((dim, dim))
    i = jnp.diag_indices(dim)
    return basis.at[i].set(jnp.e) / total




def clr(x: chex.Array) -> chex.Array:
    """ Centered log ration (clr) transform of a point on the simplex. Takes a point in the canonical basis to 
    """
    log_x = jnp.log(x)
    geom_mean = jnp.exp(jnp.mean(log_x))
    return jnp.log(x / geom_mean)

def clr_inv(x: chex.Array) -> chex.Array:
    """ The in
    """
    return jax.nn.softmax(x)


def ilr(x: chex.Array) -> chex.Array:
    """
    """
    pass

