import chex
import jax
import jax.numpy as jnp

from ssm.aitchison import clr, clr_inv, ilr_inv

"""
# Old version where the SDE is done _on_ the simplex, instead of just outside it
def make_sde(alpha: float, gamma: float, dt: float):
    
    def _sde(xt, noise):
        g_inv = riemann_inv_metric(xt)
        rescaled_noise = clr_inv(jnp.matmul(g_inv, noise))
        x_ou = apow(
            perturb(gamma, apow(xt, -1.)),
            alpha * dt
        )
        new_loc = perturb(xt, perturb(x_ou, rescaled_noise))
        return new_loc, new_loc
    
    return _sde
"""

def make_sde(alpha: chex.Scalar, gamma: chex.Array, dt: chex.Scalar):
    gamma_rn = clr(gamma)
    def _sde(xt, noise):
        x_ou = alpha * dt * (gamma_rn - clr(xt))
        new_loc = clr_inv(clr(xt) + x_ou + ilr_inv(noise))
        return new_loc, new_loc
    return _sde


def get_random_walk(key, num_steps, start, alpha, dt):
    euc_dim = start.size - 1
    rand_increments = jax.random.normal(key, (num_steps, euc_dim))*jnp.sqrt(dt)
    sde = make_sde(alpha, start, dt)
    _, path = jax.lax.scan(sde, start, rand_increments)
    return path