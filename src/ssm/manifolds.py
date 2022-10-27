import haiku as hk
import jax
import jax.numpy as jnp
import math

EPSILON = 1e-6
COS_TAYLOR_COEFFS = [
    1.0,
    -1.0 / math.factorial(2),
    +1.0 / math.factorial(4),
    -1.0 / math.factorial(6),
    +1.0 / math.factorial(8),
]
SINC_TAYLOR_COEFFS = [
    1.0,
    -1.0 / math.factorial(3),
    +1.0 / math.factorial(5),
    -1.0 / math.factorial(7),
    +1.0 / math.factorial(9),
]



def inverse_sinc(point):
    tol = EPSILON
    taylor_coeffs = jnp.array(
        [1, 1.0 / 6.0, 7.0 / 360.0, 31.0 / 15120.0, 127.0 / 604800.0]
    )
    approx = jnp.einsum(
        "k,k...->...",
        taylor_coeffs,
        jnp.float_power(taylor_coeffs, jnp.arange(5)),
    )
    point_ = jnp.sqrt(jnp.where(jnp.abs(point) <= tol, tol, point))
    exact = point_/ jnp.sin(point_)
    result = jnp.where(jnp.abs(point) < tol, approx, exact)
    return result


def inverse_tanc(point):
    tol = EPSILON
    taylor_coeffs = jnp.array(
        [1.0, -1.0 / 3.0, -1.0 / 45.0, -2.0 / 945.0, -1.0 / 4725.0]
    )
    approx = jnp.einsum(
        "k,k...->...",
        taylor_coeffs,
        jnp.float_power(taylor_coeffs, jnp.arange(5)),
    )
    point_ = jnp.sqrt(jnp.where(jnp.abs(point) <= tol, tol, point))
    exact = point_ / jnp.tan(point)
    result = jnp.where(jnp.abs(point) < tol, approx, exact)
    return result


def squared_norm(x, axis=-1, keepdims=False):
    return jnp.sum(x**2, axis=axis, keepdims=keepdims)


class Hypersphere(hk.Module):

    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.embedding_dim = dim + 1
        self.c = 1.

    @property
    def injectivity_radius(self):
        return jnp.pi / jnp.sqrt(self.c)

    def to_tangent(self, vector, base_point):
        inner_prod = jnp.dot(base_point, vector)
        coef = inner_prod / squared_norm(base_point)
        tangent_vec = vector - jnp.einsum("...,...j->...j", coef, base_point)
        return tangent_vec

    def random_uniform(self, n_samples=1):
        shape = (n_samples, self.embedding_dim)
        samples = jax.random.normal(hk.next_rng_key(), shape)
        norms = squared_norm(samples**2, axis=-1, keepdims=True)
        samples = jnp.einsum("..., ...i->...i", 1 / norms, samples)
        return samples

    def exp(self, vector, base_point):
        """ exp_{p}(v)
        p is a point in the manifold, v is a vector in the tangent space at p
        exp_{p}(v) maps to _another point on the manifold_ q
        """
        proj_tangent_vec = self.to_tangent(vector, base_point)
        norm2 = squared_norm(proj_tangent_vec)
        cos_portion = jnp.einsum("...,...j->...j", jnp.cos(norm2), base_point) 
        sinc_portion = jnp.einsum("...,...j->...j", jnp.sinc(norm2), proj_tangent_vec)
        return cos_portion + sinc_portion

    def log(self, point, base_point):
        """Compute the Riemannian logarithm of a point.

        Parameters
        ----------
        point : array-like, shape=[..., dim + 1]
            Point on the hypersphere.
        base_point : array-like, shape=[..., dim + 1]
            Point on the hypersphere.

        Returns
        -------
        log : array-like, shape=[..., dim + 1]
            Tangent vector at the base point equal to the Riemannian logarithm
            of point at the base point.
        """
        inner_prod = jnp.dot(base_point, point)
        cos_angle = jnp.clip(inner_prod, -1.0, 1.0)
        squared_angle = jnp.arccos(cos_angle) ** 2
        inv_sinc_portion = jnp.einsum("...,...j->...j",inverse_sinc(squared_angle), point)
        inv_tanc_portion = jnp.einsum("...,...j->...j", inverse_tanc(squared_angle), base_point)
        return inv_sinc_portion - inv_tanc_portion



class HypersphereProductManifold(hk.Module):

    def __init__(self, dim: int, mul: int):
        super().__init__()
        self.base_dim = dim
        self.mul = mul
        self.base_embedding_dim = dim + 1
        self.dim = dim * mul
        self.embedding_dim = (dim + 1) * mul
        self.manifold = Hypersphere(dim)
    
    def to_tangent(self, vector, base_point):
        return hk.vmap(self.manifold.to_tangent, split_rng=True)(vector, base_point)

    def random_uniform(self, n_samples=1):
        shape = (n_samples, self.mul, self.base_embedding_dim)
        samples = jax.random.normal(hk.next_rng_key(), shape)
        norms = squared_norm(samples**2, axis=-1, keepdims=True)
        return samples / norms

    def exp(self, vector, base_point):
        """ exp_{p}(v)
        """
        return hk.vmap(self.manifold.exp, split_rng=True)(vector, base_point)

    def log(self, point, base_point):
        return hk.vmap(self.manifold.log, split_rng=True)(point, base_point)



class HypersphereProductForwardGeodesicRandomWalk(hk.Module):

    def __init__(self, hypersphere_dim: int, mul: int, num_steps: int):
        super().__init__()
        self.manifold = HypersphereProductManifold(hypersphere_dim, mul)
        self.num_steps = num_steps

    def __call__(self, x0, t):
        step_size = step_size = t / self.num_steps
        gamma = jnp.sqrt(step_size)
        def _step(base_point, random_vec):
            tangent_rv = gamma * self.manifold.to_tangent(random_vec, base_point)
            point = self.manifold.exp(tangent_rv, base_point)
            return point, point
        rvs = jax.random.normal(
            hk.next_rng_key(),
            (self.num_steps, self.manifold.mul, self.manifold.base_embedding_dim)
        )
        return hk.scan(_step, x0, rvs)


def make_sudoku_walker(x0, t_final):
    manifold_random_walker = HypersphereProductForwardGeodesicRandomWalk(8, 81, 100)
    return manifold_random_walker(x0, t_final)