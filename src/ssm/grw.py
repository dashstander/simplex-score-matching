import jax
import jax.numpy as jnp


def make_forward_geodesic_random_walk_fn(manifold, num_steps):
    tangent_dim = manifold.embedding_space.dim
    def grw_fn(x0, time, rng_key):
        step_size = time / num_steps
        gamma = jnp.sqrt(step_size)
        def step(carry, rv):
            tangent_rv = gamma * manifold.to_tangent(rv, carry)
            x_new = manifold.exp(tangent_rv, carry)
            return x_new, x_new
        rvs = jax.random.normal(rng_key, (num_steps, tangent_dim))
        xt, _ = jax.lax.scan(step, x0, rvs)
        return xt
    return grw_fn

