import jax
from jax.nn import softmax
import jax.numpy as jnp
import ssm.aitchison as aitch
from ssm.utils import t_to_alpha_sigma


def kl(p, q, eps: float = 2 ** -16):
    return p.dot(jnp.log(p + eps) - jnp.log(q + eps))


def jsd(p, q, eps: float = 2. ** -16):
    m = (p + q) / 2
    pm = kl(p, m, eps)
    qm = kl(q, m, eps)
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


def kl_denoising(params, state, x0, xt, key):
    keys = jax.random.split(key, 3)
    batch_dim, seq_len, simplex_dim = x0.shape
    t = jax.random.uniform(keys[0], (batch_dim,))
    score = state.apply({'params': params},  xt, t, rngs={'dropout': keys[0]})
    score_norm = jnp.power(score, 2).sum()
    kl_div = kl(softmax(x0), softmax(score))
    return kl_div + score_norm


def jsd_denoising(params, state, x0, xt, key):
    keys = jax.random.split(key, 3)
    batch_dim, seq_len, simplex_dim = x0.shape
    t = jax.random.uniform(keys[0], (batch_dim,))
    score = state.apply({'params': params},  xt, t, rngs={'dropout': keys[0]})
    #score_norm = jnp.power(score, 2).sum()
    js_div = jsd(softmax(x0), softmax(score))
    return js_div
