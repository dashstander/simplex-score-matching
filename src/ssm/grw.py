from flax import struct
from functools import partial
import jax
import jax.numpy as jnp


@struct.dataclass
class GeodesicRandomWalk:
    manifold = struct.field(pytree_node=False)
    tangent_dim = struct.field(pytree_node=False)

    @classmethod
    def create(cls, manifold):
        tangent_dim = manifold.embedding_space.dim
        return cls(manifold, tangent_dim)

    @partial(jax.vmap, in_axes=(None, None, 0, 0, 0))
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
