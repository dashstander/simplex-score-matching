import jax
import jax.numpy as jnp
import ssm.deprecated.aitchison as aitch
from ssm.utils import t_to_alpha_sigma



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





@jax.vmap
def kl(p, q):
    eps: float = 2 ** -16
    return jnp.tensordot(
        p,
        jnp.log(p + eps) - jnp.log(q + eps)
    )


def jsd(p, q):
    m = jnp.add(p, q) / 2
    pm = kl(p, m)
    qm = kl(q, m)
    return (pm + qm) / 2


def sliced_score_matching(params, state, xt, t, keys):
    """
    $x_{t}\in\mathbb{R}^{n}$ is in CLR (centered log-ratio) coordinates. It's one softmax away from being on the simplex.
    """
    # v as defined in Sliced Score Matching
    v = jax.random.normal(keys[0], xt.shape)
    def score_fn(x):
        return state.apply_fn({'params': params}, x, t, rngs={'dropout': keys[1]})
    # Jacobian vector product with random vector $v ~ N(0, I)$ approximates $tr(J)$
    # Since the score is already approximating the gradient of the data, the Jacobian of the score is the Hessian of the data
    # Using forward mode (jvp) instead of reverse (vjp) because I _think_ that it performs better, since we use reverse in the 
    # outer optimization loop to do backprop
    # TODO: Test this
    score, score_grad_dot_v = jax.jvp(
        score_fn,
        (xt,),
        (v,)
    )
    score_norm = jnp.power(score, 2).sum()
    hutchinson_est = jax.vmap(jnp.tensordot)(v, score_grad_dot_v).sum()
    return 0.5 * score_norm + hutchinson_est


def v_denoising(params, state, x0, key):
    """
    `x` is in $R^{n} in CLR (centered log-ratio) coordinates. It's one softmax away from being on the simplex.
    """
    keys = jax.random.split(key, 3)
    batch_dim, seq_len, simplex_dim = x0.shape
    t = jax.random.uniform(keys[0], (batch_dim,))
    alphas, sigmas = t_to_alpha_sigma(t)
    alphas, sigmas = jnp.expand_dims(alphas, (1, 2)), jnp.expand_dims(sigmas, (1, 2))
    # Generate noise
    raw_noise = jax.random.normal(keys[1], (batch_dim, seq_len, simplex_dim))
    x = aitch.clr(x0, axis=-1, keepdims=True)
    simplex_scaled_noise = aitch.simplex_metric_tensor_inv(
        x0,
        raw_noise
    )
    noised_x = alphas * x + sigmas * simplex_scaled_noise
    targets = alphas * simplex_scaled_noise - sigmas * x
    v = state.apply_fn({'params': params}, noised_x, t, rngs={'dropout': keys[3]})
    return jnp.mean((v - targets)**2)




def grw_denoising(params, model_fn, forward_noise_fn, x0, key):
    keys = jax.random.split(key, 3)
    batch_dim, seq_len, simplex_dim = x0.shape
    t = jax.random.uniform(keys[0], (batch_dim,))

    xt = forward_noise_fn(x0, t)
    pred_flow = model_fn(params, keys[1], xt, t)

