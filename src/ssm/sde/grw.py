import jax
import jax.numpy as jnp
import treeo as to

from ssm.manifolds import HypersphereProductManifold



class HypersphereProductForwardGeodesicRandomWalk(to.Tree):
    manifold: HypersphereProductManifold = to.field(node=False)
    num_steps: int = to.field(default=100, node=False)
    beta_0: float = to.field(default=0.1, node=False)
    beta_f: float = to.field(default=5., node=False)

    def __init__(self, hypersphere_dim: int, mul: int, num_steps: int, beta_0: float, beta_f: float):
        self.manifold = HypersphereProductManifold(hypersphere_dim - 1, mul)
        self.num_steps = num_steps
        self.beta_0 = beta_0
        self.beta_f = beta_f

    @property
    def shape_extrinsic(self):
        return (self.manifold.mul, self.manifold.base_embedding_dim)

    def beta_t(self, t):
        #normed_t = (t - self.t0) / (self.tf - self.t0)
        return self.beta_0 + t * (self.beta_f - self.beta_0)

    def rescale_t(self, t):
        return 0.5 * t**2 * (self.beta_f - self.beta_0) + t * self.beta_0

    def grad_marginal_log_prob(self, x0, x, t):
        logp_grad = self.manifold.grad_log_heat_kernel(x, x0, t, jnp.array(0.))
        return logp_grad

    def rw(self, x0, t, rng):
        step_size = step_size = t / self.num_steps
        gamma = jnp.sqrt(step_size)
        def _step(base_point, data):
            noise, t = data
            sigma_t = jnp.sqrt(self.beta_t(t))
            tangent_rv = sigma_t * gamma * self.manifold.to_tangent(noise, base_point)
            point = self.manifold.exp(tangent_rv, base_point)
            return point, point
        random_vecs = jax.random.normal(rng, (self.num_steps, *self.shape_extrinsic))
        times = jnp.linspace(0., t, self.num_steps)
        return jax.lax.scan(_step, x0, (random_vecs, times))


class HypersphereProductBackwardGeodesicRandomWalk(to.Tree):

    def __init__(self, dim: int, mul: int, num_steps: int, score_fn, beta_0, beta_f):
        super().__init__()
        self.manifold = HypersphereProductManifold(dim - 1, mul)
        self.num_steps = num_steps
        self.score_fn = score_fn
        self.beta_0 = beta_0
        self.beta_f = beta_f
    
    @property
    def shape_extrinsic(self):
        return (self.manifold.mul, self.manifold.base_embedding_dim)

    def beta_t(self, t):
        #normed_t = (t - self.t0) / (self.tf - self.t0)
        return self.beta_0 + t * (self.beta_f - self.beta_0)

    def __call__(self, x_final, mask, t_final, rng):
        step_size = step_size = t_final / self.num_steps
        gamma = jnp.sqrt(step_size)
        def _step(base_point, data):
            random_vec, t, key = data
            sigma_t = jnp.sqrt(self.beta_t(t))
            tangent_rv = sigma_t * gamma * self.manifold.to_tangent(random_vec, base_point)
            drift_term = step_size * self.score_fn(key, base_point, mask, t)
            drift_term = jnp.squeeze(drift_term)
            point = self.manifold.exp(drift_term + tangent_rv, base_point)
            point = jnp.abs(point)
            return point, point
        times = jnp.linspace(t_final, 0., self.num_steps)
        rvkey, subkey = jax.random.split(rng)
        keys = jax.random.split(subkey, self.num_steps)
        random_vecs = jax.random.normal(rvkey, (self.num_steps, *self.shape_extrinsic))
        x0, path = jax.lax.scan(_step, x_final, (random_vecs, times, keys))
        return x0, path
