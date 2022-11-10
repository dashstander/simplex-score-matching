from dataclasses import dataclass
from einops import rearrange, repeat
from functools import partial
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np


@dataclass
class TransformerConfig:
    vocab_size: int
    embed_dim: int
    model_dim: int
    mlp_dim: int
    num_layers: int = 3
    time_dim: int = 16
    num_heads: int = 8
    max_length: int = 512
    dropout: float = 0.1
    attention_dropout: float = 0.1
    fourier_init_std: float = 0.2


LayerNorm = partial(hk.LayerNorm, create_scale=True, create_offset=True, axis=-1)


def squared_norm(x, axis=-1, keepdims=False):
    return jnp.sum(x**2, axis=axis, keepdims=keepdims)


def to_tangent(vector, base_point):
    #inner_prod = jax.vmap(jax.vmap(jnp.dot))(base_point, vector)
    inner_prod = jnp.einsum("...j,...j->...", vector, base_point)
    coef = inner_prod / squared_norm(base_point)
    tangent_vec = vector - jnp.einsum("...,...j->...j", coef, base_point)
    return tangent_vec



def normalize(x, axis=-1, keepdims=True):
    norms = squared_norm(x, axis=axis, keepdims=keepdims)
    normalized = x / norms
    return jnp.nan_to_num(normalized)

def rotate_every_two(x):
    x1 = x[:, :, :, 0::2]
    x2 = x[:, :, :, 1::2]
    x = jnp.stack((-x2, x1), axis=-1)
    return rearrange(x, '... d j -> ... (d j)')


def apply_rotary_pos_emb(x, sincos):
    sin, cos = map(lambda t: repeat(t, '... b n -> ... b (n j)', j=2)[-x.shape[-3]:, None, :], sincos)
    return (x * cos) + (rotate_every_two(x) * sin)


def fixed_pos_embedding(x, seq_dim: int = 0):
    dim = x.shape[-1]
    inv_freq = 1. / (10000 ** (np.arange(0, dim, 2) / dim))
    sinusoid_inp = np.einsum('i , j -> i j', np.arange(x.shape[seq_dim]), inv_freq)
    return np.sin(sinusoid_inp), np.cos(sinusoid_inp)



class FourierFeatures(hk.Module):
    def __init__(self, output_size: int = 16, std: float = 1., name: str = None):
        super().__init__(name=name)
        assert output_size % 2 == 0
        self.output_size = output_size
        self.std = std

    def __call__(self, x):
        w = hk.get_parameter(
            'w',
            [self.output_size // 2,
            x.shape[1]],
            init=hk.initializers.RandomNormal(self.std, 0)
        )
        f = 2 * jnp.pi * x @ w.T
        return jnp.concatenate([jnp.cos(f), jnp.sin(f)], axis=-1)


class AttentionBlock(hk.Module):

    def __init__(self, config: TransformerConfig, name: str = None):
        super().__init__(name=name)
        self.config = config
        self.ln = LayerNorm()
        self.mha = hk.MultiHeadAttention(
            self.config.num_heads,
            self.config.model_dim,
            w_init = hk.initializers.RandomNormal()
        )

        self.dropout_rate = self.config.attention_dropout

    def __call__(self, x, dropout: float = 1.):
        x = hk.dropout(hk.next_rng_key(), dropout * self.dropout_rate, x)
        x = self.ln(x)
        return self.mha(x, x, x)


class FeedForward(hk.Module):

    def __init__(self, config: TransformerConfig, name: str = None):
        super().__init__(name=name)
        self.config = config
        self.ln = LayerNorm()
        self.dense_in = hk.Linear(self.config.mlp_dim, with_bias=False)
        self.dense_out = hk.Linear(self.config.model_dim, with_bias=False)
        self.dropout_rate = self.config.dropout

    def __call__(self, x, dropout: float = 1.):
        x = hk.dropout(
            hk.next_rng_key(),
            dropout * self.dropout_rate,
            x # TODO: somehow _this_ is getting passed in as a TransformerConfig
        )
        x = self.ln(x)
        x = self.dense_in(x)
        x = jax.nn.relu(x)
        x = self.dense_out(x)
        return x


class TransformerLayer(hk.Module):

    def __init__(self, config: TransformerConfig, name: str = None):
        super().__init__(name=name)
        self.transformer = AttentionBlock(config)
        self.ff = FeedForward(config)

    def __call__(self, x0, dropout: float = 1.):
        #x_rot = x[:, :, :self.d_rotary]
        #x_pass = x[ :, :, self.d_rotary:]
        #sincos = fixed_pos_embedding(x_rot, seq_dim=1)
        #x_rot = apply_rotary_pos_emb(x_rot, sincos)
        #x = jnp.concatenate([x_rot, x_pass], axis=-1)
        x = self.transformer(x0, dropout)
        return x0 + self.ff(x)


class TransformerDiffusion(hk.Module):

    def __init__(self, config: TransformerConfig, is_training: bool = True, name: str = None):
        super().__init__(name=name)
        self.config = config
        self.dropout = 1. if is_training else 0.
        self.linear0 = hk.Linear(self.config.embed_dim)
        self.fourier = FourierFeatures(self.config.time_dim)
        self.ff1 = FeedForward(config)
        self.transformers = [TransformerLayer(self.config) for _ in range(self.config.num_layers)]
        self.linear1 = hk.Linear(self.config.vocab_size)
        self.linear2 = hk.Linear(self.config.vocab_size)

    def __call__(self, xt, mask, t):
        x = self.linear0(xt + mask)
        timestep_embed = self.fourier(t[:, None])
        te_planes = jnp.tile(timestep_embed[:, None], (1, self.config.max_length, 1))
        x = jnp.concatenate([x, te_planes], axis=-1)
        x = self.ff1(x, self.dropout)
        trans_x = x
        for layer in self.transformers:
            trans_x = layer(trans_x, self.dropout)
        x = x + jnp.sqrt(2) * trans_x
        x = self.linear1(x) + xt
        x = self.linear2(x) + mask
        vf = to_tangent(x, xt)
        return vf


def make_diffusion_fn(model_config, training):
    def fn(x, mask, t):
        pred = TransformerDiffusion(model_config, training)(x, mask, t)
        return pred
    return fn
