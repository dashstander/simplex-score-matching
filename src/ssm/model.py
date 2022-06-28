from einops import rearrange, repeat
from flax import linen as nn
from flax import struct
import jax.numpy as jnp
import numpy as np
from typing import Sequence

from ssm.aitchison import clr
from ssm.utils import alpha_sigma_to_log_snr, t_to_alpha_sigma


@struct.dataclass
class TransformerConfig:
    vocab_size: int
    embed_dim: int
    model_dim: int
    mlp_dim: int
    num_layers: int = 3
    time_dim: int = 16
    num_heads: int = 8
    max_length: int = 512
    dropout_rate: float = 0.1
    attention_dropout_rate: float = 0.1
    # kernel_init: Callable = nn.initializers.xavier_uniform()
    fourier_init_std: float = 0.2


def normalize_probabilities(x):
    logx = jnp.log1p(x)
    x_mean0 = logx - jnp.mean(logx, axis=-1, keepdims=True)
    x_normalized = x_mean0 / jnp.var(x_mean0, axis=-1, keepdims=True)
    return x_normalized


def fixed_pos_embedding(x, seq_dim=0):
    dim = x.shape[-1]
    inv_freq = 1.0 / (10000 ** (np.arange(0, dim, 2) / dim))

    sinusoid_inp = np.einsum("i , j -> i j", np.arange(x.shape[seq_dim]), inv_freq)

    return np.sin(sinusoid_inp), np.cos(sinusoid_inp)


def rotate_every_two(x):
    """ Rotary embeddings,
    N.B: The layout of the queries and keys is [seq, n_head, d_head] (no batch dim).
    """
    x1 = x[:, :, ::2]
    x2 = x[:, :, 1::2]
    x = jnp.stack((-x2, x1), axis=-1)
    return rearrange(x, "... d j -> ... (d j)")


def apply_rotary_pos_emb(x, sincos):
    """ Rotary embeddings,
    N.B: The layout of the queries and keys is [seq, n_head, d_head] (no batch dim).
    """
    sin, cos = map(lambda t: repeat(t, "b n -> b (n j)", j=2)[:, None, :], sincos)
    return (x * cos) + (rotate_every_two(x) * sin)



"""
class RMSNorm(nn.Module):

    @nn.compact
    def __call__(self, x):
        eps = 1e-8
        dim = x.shape[-1]
        scale = jnp.power(dim, -0.5)
        g = self.param('g', nn.initializers.ones, (dim,))
        norm = jax.nn.normalize(x, axis=-1) * scale
        return x / jax.lax.clamp(eps, norm) * g
"""



class FourierFeatures(nn.Module):
    config: TransformerConfig

    @nn.compact
    def __call__(self, x):
        w = self.param(
            'w',
            nn.initializers.normal(stddev=self.config.fourier_init_std),
            (self.config.time_dim // 2, x.shape[1]),
        )
        f = 2 * jnp.pi * x @ w.T
        return jnp.concatenate([jnp.cos(f), jnp.sin(f)], axis=-1)


class SelfAttention(nn.Module):
    config: TransformerConfig

    @nn.compact
    def __call__(self, x, padding_mask=None, deterministic=False):
        x = nn.LayerNorm(use_bias=False, use_scale=False)(x)
        padding_mask = None if padding_mask is None else padding_mask[:, None, None, :]
        x = nn.MultiHeadDotProductAttention(
            self.config.num_heads,
            dropout_rate=self.config.attention_dropout_rate
        )(x, x, padding_mask, deterministic=deterministic)
        return x


class FeedForward(nn.Module):
    config: TransformerConfig

    @nn.compact
    def __call__(self, x):
        x = nn.LayerNorm(use_bias=False, use_scale=False)(x)
        x = nn.Dense(self.config.mlp_dim, use_bias=False)(x)
        x = nn.gelu(x)
        x = nn.Dense(self.config.model_dim, use_bias=False)(x)
        return x


class TransformerLayer(nn.Module):
    config: TransformerConfig

    @nn.compact    
    def __call__(self, x, padding_mask=None, deterministic=False):
        #x_rot = x[:, :, :self.d_rotary]
        #x_pass = x[ :, :, self.d_rotary:]
        #sincos = fixed_pos_embedding(x_rot, seq_dim=1)
        #x_rot = apply_rotary_pos_emb(x_rot, sincos)
        #x = jnp.concatenate([x_rot, x_pass], axis=-1)
        x = x + SelfAttention(self.config)(x, padding_mask, deterministic)
        x = x + FeedForward(self.config)(x)
        return x


class SkipBlock(nn.Module):
    layers: Sequence[nn.Module]

    @nn.compact
    def __call__(self, x):
        x_new = nn.Sequential(*self.layers)(x)
        return jnp.concatenate(
            [x_new, x], 
            dim=1
        )


class TransformerDiffusion(nn.Module):
    config: TransformerConfig
    
    @nn.compact
    def __call__(self, x, t, training=False):
        """
        x.shape = 
        """
        deterministic = not training
        x = normalize_probabilities(x)
        x_init = nn.Dense(self.config.embed_dim)(x)
        log_snr = alpha_sigma_to_log_snr(*t_to_alpha_sigma(t))
        timestep_embed = FourierFeatures(self.config)(log_snr[:, None])
        te_planes = jnp.tile(timestep_embed[:, None], (1, self.config.max_length, 1))
        x = jnp.concatenate([x_init, te_planes], axis=-1)
        x = FeedForward(self.config)(x)
        trans_x = nn.Sequential([
            TransformerLayer(self.config) for _ in range(self.config.num_layers)
        ])(x, None, deterministic=deterministic)        
        x = x + jnp.sqrt(2) * trans_x
        x_final = nn.Dense(self.config.vocab_size)(x)
        return clr(x_final, axis=-1, keepdims=True)
