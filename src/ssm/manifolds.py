import geomstats.algebra_utils as gsutils
import haiku as hk
import jax
import jax.numpy as jnp
import math


def squared_norm(x, axis=-1, keepdims=False):
    return jnp.sum(x**2, axis=axis, keepdims=keepdims)


def make_gegenbauer_polynomials(alpha, l_max):

    def out_fn(x):

        def step(carry, n):
            C_nm2, C_nm1 = carry
            C_n = (
                1 / n * (2 * x * (n + alpha - 1) * C_nm1 - 
                (n + 2 * alpha - 2) * C_nm2)
            )
            return (C_nm1, C_n), C_n

        C_0 = jnp.ones_like(x)
        C_1 = 2 * alpha * x
        initial_polys = jnp.array([C_0, C_1])
        _, polys = jax.lax.scan(
            step,
            (C_0, C_1),
            jnp.arange(2, l_max + 1),
        )
        all_polys = jnp.concatenate([initial_polys, polys])
        return all_polys
    
    return out_fn


class Hypersphere(hk.Module):

    def __init__(self, dim, gegenbaur_num_approx=20):
        super().__init__()
        self.dim = dim
        self.embedding_dim = dim + 1
        self.c = 1.
        self.nmax = gegenbaur_num_approx
        self.gegenbaur_poly_fn = make_gegenbauer_polynomials((dim - 1) / 2, self.nmax)

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
        coef_1 = gsutils.taylor_exp_even_func(norm2, gsutils.cos_close_0, order=4)
        coef_2 = gsutils.taylor_exp_even_func(norm2, gsutils.sinc_close_0, order=4)
        exp = jnp.einsum("...,...j->...j", coef_1, base_point) + jnp.einsum(
            "...,...j->...j", coef_2, proj_tangent_vec
        )
        return exp

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
        coef_1_ = gsutils.taylor_exp_even_func(
            squared_angle, gsutils.inv_sinc_close_0, order=5
        )
        coef_2_ = gsutils.taylor_exp_even_func(
            squared_angle, gsutils.inv_tanc_close_0, order=5
        )
        log = jnp.einsum("...,...j->...j", coef_1_, point) - jnp.einsum(
            "...,...j->...j", coef_2_, base_point
        )
        return log

    @property
    def surface_area(self):
        half_dim = self.dim / 2
        return 2 * jnp.pi**half_dim / math.gamma(half_dim)
    
    def heat_kernel_poly_approx(self, x, x0, t):
        """
        log p_t(x, y) = \sum^\infty_n e^{-t \lambda_n} \psi_n(x) \psi_n(y)
        = \sum^\infty_n e^{-n(n+1)t} \frac{2n+d-1}{d-1} \frac{1}{A_{\mathbb{S}^n}} \mathcal{C}_n^{(d-1)/2}(x \cdot y
        """
        # NOTE: in the original code, divided t by 2 to "match random walk"
        d = self.dim
        
        n = jnp.arange(0, self.nmax + 1)[..., None]
        t = jnp.expand_dims(t, axis=0)

        # TODO: This is an sequence that gets small very very rapidly, the polynomials get _large_ very rapidly,
        # how to best maintain precision?
        coeffs = (
            jnp.exp(-n * (n + 1) * t) * (2 * n + d - 1) / (d - 1) / self.surface_area
        )
        inner_prod = jnp.sum(x0 * x, axis=-1)
        cos_theta = jnp.clip(inner_prod, -1.0, 1.0)
        P_n = self.gegenbaur_poly_fn(cos_theta)
        prob = jnp.sum(coeffs * P_n, axis=0)
        print(prob)
        return jnp.log(prob)

    def heat_kernel_varadhan_approx(self, xt, xs, t, s):
        """ t > s"""
        delta_t = t - s
        #axis_to_expand = tuple(range(-1, -len(xt.shape), -1))  # (-1) or (-1, -2)
        #delta_t = jnp.expand_dims(delta_t, axis=axis_to_expand)
        grad = self.log(xs, xt) / delta_t
        return grad


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

    def grad_log_heat_kernel(self, xt, xs, t, s):
        grad_heat_kernel = hk.vmap(
            self.manifold.heat_kernel_varadhan_approx,
            in_axes=(0, 0, None, None),
            split_rng=True
        )
        return grad_heat_kernel(xt, xs, t, s)

    def log(self, point, base_point):
        return hk.vmap(self.manifold.log, split_rng=True)(point, base_point)
