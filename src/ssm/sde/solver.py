import jax
import jax.numpy as jnp
import treeo as to

from ssm.manifolds import HypersphereProductManifold


class HypersphereBackwardsSolver(to.Tree):
    manifold: HypersphereProductManifold = to.field(node=False)
    num_steps: int = to.field(default=100, node=False)
    cfg_weight: float = to.field(default=5., node=False)
    score_model = to.field(node=False)

    def __init__(self, dim: int, mul: int, num_steps: int, cfg_weight: float, score_model):
        self.manifold = HypersphereProductManifold(dim - 1, mul)
        self.num_steps = num_steps
        self.score_model = score_model
    
    @property
    def shape_extrinsic(self):
        return (self.manifold.mul, self.manifold.base_embedding_dim)

    def score_fn(self, params, rng, x, mask, time):
        k1, k2 = jax.random.split(rng)
        uncond_mask = jnp.zeros_like(mask)
        # crowsonkb: i normally use the uncond-centered version which is w * eps(z, c) + (1 - w) * eps(z).  or uncond_score + w * (cond_score - uncond_score).
        # so w=0 means uncond, w=1 means cond, -1 means negative cond, etc
        uncond_score = self.score_model.apply(params, k1, x, uncond_mask, time)
        cond_score = self.score_model.apply(params, k2, x, mask, time)
        return self.cfg_weight * (cond_score) + (1 - self.cfg_weight) * uncond_score

    def solve(self, params, rng, x_final, mask, t_final):
        step_size = step_size = t_final / self.num_steps
        def _step(base_point, data):
            key, t = data
            k1, k2 = jax.random.split(key)
            batch_size = jnp.shape(base_point)[0]
            times = jnp.full((batch_size,), t)
            logits = self.score_fn(params, k1, base_point, mask, times)
            pred = jax.random.categorical(k2, logits)
            drift_term = step_size * jax.vmap(self.manifold.log)(pred, base_point)
            point = jax.vmap(self.manifold.exp)(drift_term, base_point)
            point = jnp.abs(point)
            return point, point
        times = jnp.linspace(t_final, 0., self.num_steps)
        keys = jax.random.split(rng, self.num_steps)
        x0, path = jax.lax.scan(_step, x_final, (keys, times))
        return x0, path
