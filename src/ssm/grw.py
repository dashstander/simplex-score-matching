from functools import partial
import jax
import jax.numpy as jnp


class ForwardGeodesicRandomWalk:

    def __init__(self, manifold):
        self.manifold = manifold
        self.tangent_dim = manifold.embedding_space.dim

    #@partial(jax.vmap, in_axes=(None, None, 0, 0, 0))
    def geodesic_random_walk(self, num_steps, time, rng_key, x0=None):
        step_size = time / num_steps
        gamma = jnp.sqrt(step_size)
        if x0 is None:
            rng_key, x0_key = jax.random.split(rng_key)
            x0 = self.manifold.random_uniform(state=x0_key)
        def grw_step(carry, rv):
            tangent_rv = gamma * self.manifold.to_tangent(rv, carry)
            x_new = self.manifold.exp(tangent_rv, carry)
            return x_new, x_new
        rvs = jax.random.normal(rng_key, (num_steps, self.tangent_dim))
        _, path = jax.lax.scan(grw_step, x0, rvs)
        return path
