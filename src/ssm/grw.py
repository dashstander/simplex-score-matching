import haiku as hk
import jax
import jax.numpy as jnp

from ssm.manifolds import HypersphereProductManifold


class HypersphereProductForwardGeodesicRandomWalk(hk.Module):

    def __init__(self, hypersphere_dim: int, mul: int, num_steps: int, beta_0: float, beta_f: float):
        super().__init__()
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

    def __call__(self, x0, t):
        step_size = step_size = t / self.num_steps
        gamma = jnp.sqrt(step_size)
        hk.reserve_rng_keys(self.num_steps)
        def _step(base_point, t):
            noise = jax.random.normal(hk.next_rng_key(), self.shape_extrinsic)
            sigma_t = jnp.sqrt(self.beta_t(t))
            tangent_rv = sigma_t * gamma * self.manifold.to_tangent(noise, base_point)
            point = self.manifold.exp(tangent_rv, base_point)
            return point, point
        times = jnp.linspace(0., t, self.num_steps)
        return hk.scan(_step, x0, times)


class HypersphereProductBackwardGeodesicRandomWalk(hk.Module):

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

    def __call__(self, x_final, mask, t_final):
        step_size = step_size = t_final / self.num_steps
        gamma = jnp.sqrt(step_size)
        hk.reserve_rng_keys(self.num_steps * 2)
        def _step(base_point, t):
            random_vec = jax.random.normal(hk.next_rng_key(), self.shape_extrinsic)
            sigma_t = jnp.sqrt(self.beta_t(t))
            tangent_rv = sigma_t * gamma * self.manifold.to_tangent(random_vec, base_point)
            drift_term = step_size * self.score_fn(hk.next_rng_key(), base_point, mask, t)
            drift_term = jnp.squeeze(drift_term)
            point = self.manifold.exp(drift_term + tangent_rv, base_point)
            #point = jnp.where(mask, base_point, point)
            point = jnp.abs(point)
            return point, point
        times = jnp.linspace(t_final, 0., self.num_steps)
        x0, path = hk.scan(_step, x_final, times)
        return x0, path


def make_sudoku_forward_walker(num_steps, beta_0, beta_f):
    def walker(x0, t_final):
        manifold_random_walker = HypersphereProductForwardGeodesicRandomWalk(9, 81, num_steps, beta_0, beta_f)
        xt, _ = manifold_random_walker(x0, t_final)
        grad_log_prob = manifold_random_walker.grad_marginal_log_prob(x0, xt, t_final)
        return jnp.abs(xt), grad_log_prob
    return hk.vmap(walker, split_rng=True)


def make_sudoku_solver(score_fn, num_steps, beta_0, beta_f):
   
    def walker(x_final, mask, t_final):
        manifold_rw = (
            HypersphereProductBackwardGeodesicRandomWalk(9, 81, num_steps, score_fn, beta_0, beta_f)
        )
        x0, _ = manifold_rw(x_final, mask, t_final)
        return x0
    return hk.vmap(walker, split_rng=True)
