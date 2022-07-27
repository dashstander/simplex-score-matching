from diffrax import diffeqsolve, Euler, MultiTerm, ODETerm, VirtualBrownianTree
from diffrax.term import _ControlTerm
from functools import partial
import jax
import jax.numpy as jnp

import ssm.aitchison as aitch



class SimplexControlTerm(_ControlTerm):

    @staticmethod
    def prod(vf, control):
        _, g_inv = jax.jvp(
            jax.nn.softmax,
            (vf, ),
            (control, )
        )
        return jnp.sqrt(2) * g_inv



def dirichlet_potential(p, alphas):
    num_alphas = alphas.sum()
    return (num_alphas * p) - alphas


def drift_potential(t, x, args):
    x = x / jnp.var(x, axis=-1, keepdims=True)
    p = jax.nn.softmax(x)
    alphas = jnp.ones((p.shape[-1],))
    return -1 * aitch.simplex_metric_tensor_inv(
        p,
        dirichlet_potential(p, alphas)
    )


def softmax_jac_squared(x):
    _jvp = lambda s: jax.jvp(jax.nn.softmax, (x,), (s,))[1]
    return jax.vmap(_jvp)(jax.jacrev(jax.nn.softmax)(x))


def reverse_drift(model_fn, t, x, args):
    drift = drift_potential(t, x, {})
    score = model_fn(x, t)
    j_sfm = softmax_jac_squared(x)
    return drift - (0.5 * jnp.matmul(j_sfm, score))


@jax.vmap
def dirichlet_forward_sde(x0, t1, key):
    """ The forward noising SDE that converges to Dir([1, 1, ...., 1]) (i.e. uniform on the simplex)
    """
    t0 = 0
    brownian_motion = VirtualBrownianTree(t0, t1, tol=0.05, shape=x0.shape, key=key)
    terms = MultiTerm(ODETerm(drift_potential), SimplexControlTerm(lambda t, y, args: y, brownian_motion))
    solver = Euler() 
    sol = diffeqsolve(terms, solver, t0, t1, dt0=0.1, y0=x0)
    return sol.ys[0]


@partial(jax.vmap, in_axes=(None, None, 0, 0))
def dirichlet_reverse_sde(model, size, t1, key):

    def drift(t, x, args):
        return reverse_drift(model, t, x, args)

    t0 = 1e-8
    init_key, dw_key = jax.random.split(key)
    x1 = jax.random.dirichlet(init_key, jnp.ones(size))
    brownian_motion = VirtualBrownianTree(t1, t0, tol=0.05, shape=x1.shape, key=dw_key)
    terms = MultiTerm(ODETerm(drift), SimplexControlTerm(brownian_motion))
    solver = Euler()
    sol = diffeqsolve(terms, solver, t1, t0, dt0=-0.1, y0=aitch.clr(x1, axis=-1, keepdims=True))
    return sol.ys[0]

