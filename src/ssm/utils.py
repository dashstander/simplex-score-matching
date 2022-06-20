import chex
import jax
import jax.numpy as jnp
import jax.random as jrand
import numpy as np

""" Signal / Noise ratio utils stolen shamelessly from @crowsonkb,
https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/utils.py
"""


def log_snr_to_alpha_sigma(log_snr):
    """Returns the scaling factors for the clean image and for the noise, given
    the log SNR for a timestep."""
    return jnp.sqrt(jax.nn.sigmoid(log_snr)), jnp.sqrt(jax.nn.sigmoid(-log_snr))


def alpha_sigma_to_log_snr(alpha, sigma):
    """Returns a log snr, given the scaling factors for the clean image and for
    the noise."""
    return jnp.log(alpha**2 / sigma**2)


def t_to_alpha_sigma(t):
    """Returns the scaling factors for the clean image and for the noise, given
    a timestep."""
    return jnp.cos(t * jnp.pi / 2), jnp.sin(t * jnp.pi / 2)


def alpha_sigma_to_t(alpha, sigma):
    """Returns a timestep, given the scaling factors for the clean image and for
    the noise."""
    return jnp.arctan2(sigma, alpha) / jnp.pi * 2


def get_ddpm_schedule(ddpm_t):
    """Returns timesteps for the noise schedule from the DDPM paper."""
    log_snr = -jnp.log(jnp.expm1(1e-4 + 10 * ddpm_t**2))
    alpha, sigma = log_snr_to_alpha_sigma(log_snr)
    return alpha_sigma_to_t(alpha, sigma)



def unreplicate(x):
    return jax.tree_map(lambda x: x[0], x)


def psplit(x, n):
    return jax.tree_map(lambda x: jnp.stack(jnp.split(x, n)), x)


def punsplit(x):
    return jax.tree_map(
        lambda x: jnp.reshape(x, (x.shape[0] * x.shape[1], *x.shape[2:])),
        x
    )



def tokens_to_probs(rng, token_ids, concentration=0.9, vocab_size=8192):
    """ Takes a 
    """
    batch, seq_len = token_ids.shape
    
    def _tokens_to_probs(token_ids):
        x = rng.random((seq_len, vocab_size)) / vocab_size
        # At this point E(x.sum()) == 0.5 
        # What we want is for new_val / (x.sum() + new_val) ~ concentration
        # --> new_val == (concentration * x.sum())/(1 - concentration)
        # Then, in the normalized vector, the appropriate token will have ~ concentration weight,
        # and the others will have the rest
        x_sum = x.sum(axis=1)
        conc_val = np.mean((concentration * x_sum) / (1 - concentration))
        np.put_along_axis(x, token_ids[:, None], conc_val, axis=1)
        return x / x.sum(axis=1)[:, None]
    return np.apply_along_axis(_tokens_to_probs, axis=1, arr=token_ids)



def probs_to_tokens(key, tokenizer, token_probs: chex.Array):
    """
    """
    seq_len = token_probs.shape[0]
    sample_fn = lambda k, p: jrand.choice(k, tokenizer.vocab_size, 1, p=p)
    tokens = jax.lax.map(sample_fn, (jrand.split(key, seq_len), token_probs))
    return tokenizer.decode(tokens)