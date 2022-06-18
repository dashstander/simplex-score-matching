from einops import rearrange, repeat
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np

from ssm.utils import alpha_sigma_to_log_snr, t_to_alpha_sigma


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



class FourierFeatures(hk.Module):
    def __init__(self, output_size, std=1., name=None):
        super().__init__(name=name)
        assert output_size % 2 == 0
        self.output_size = output_size
        self.std = std

    def __call__(self, x):
        w = hk.get_parameter(
            'w',
            [self.output_size // 2, x.shape[1]],
            init=hk.initializers.RandomNormal(self.std, 0)
        )
        f = 2 * jnp.pi * x @ w.T
        return jnp.concatenate([jnp.cos(f), jnp.sin(f)], axis=-1)


class SelfAttention(hk.Module):
    def __init__(self, num_heads=1, name=None):
        super().__init__(name=name)
        self.num_heads = num_heads

    def __call__(self, x, padding_mask=None):
        # d_model = x.shape[-1]
        x = hk.RMSNorm(axis=-1)(x)
        padding_mask = None if padding_mask is None else padding_mask[:, None, None, :]
        x = hk.MultiHeadAttention(
            self.num_heads,
            x.shape[-1] // self.num_heads,
            1.
        )(x, x, x, padding_mask)
        return x


class FeedForward(hk.Module):
    def __init__(self, d_ff, name=None):
        super().__init__(name=name)
        self.d_ff = d_ff

    def __call__(self, x):
        d_model = x.shape[-1]
        x = hk.RMSNorm(axis=-1)(x)
        x = hk.Linear(self.d_ff, name='linear_0')(x)
        x = jax.nn.gelu(x)
        x = hk.Linear(d_model, name='linear_1')(x)
        return x


class TransformerLayer(hk.Module):
    def __init__(self, d_ff, n_heads, name=None):
        super().__init__(name=name)
        self.d_ff = d_ff
        self.n_heads = n_heads
        # self.d_rotary = d_rotary
        
    def __call__(self, x, padding_mask=None):
        #x_rot = x[:, :, :self.d_rotary]
        #x_pass = x[ :, :, self.d_rotary:]
        #sincos = fixed_pos_embedding(x_rot, seq_dim=1)
        #x_rot = apply_rotary_pos_emb(x_rot, sincos)
        #x = jnp.concatenate([x_rot, x_pass], axis=-1)
        x = x + SelfAttention(self.n_heads)(x, padding_mask)
        x = x + FeedForward(self.d_ff)(x)
        return x


class SkipBlock(hk.Module):
    def __init__(self, main, skip=None):
        super().__init__()
        self.main = hk.Sequential(*main)

    def __call__(self, x):
        return jnp.concatenate(
            [self.main(x), x], 
            dim=1
        )

class TransformerDiffusion(hk.Module):

    def __init__(self, init_embed, d_embed, time_embed, n_layers, d_model, n_heads, vocab_size, name=None):
        super().__init__(name=name)
        self.init_embed = init_embed
        self.d_embed = d_embed
        self.time_embed = time_embed
        self.n_layers = n_layers
        self.d_model = d_model
        self.d_ff = self.d_model * 4
        self.n_heads = n_heads
        self.vocab_size = vocab_size
    
    def __call__(self, x, t):
        """
        x.shape = 
        """
        x_init = hk.Linear(self.init_embed)(x)
        log_snr = alpha_sigma_to_log_snr(t_to_alpha_sigma(t))
        timestep_embed = FourierFeatures(self.time_embed, 0.2)(log_snr[:, None])
        te_planes = jnp.tile(timestep_embed[..., None], [x.shape[0], 1])
        x = jnp.concatenate([x_init, te_planes], axis=1)
        layers = hk.Sequential( * [
            TransformerLayer(
                self.d_ff,
                self.n_heads,
                name=f'transformer_{i}'
            ) for i in range(self.n_layers)
        ])
        x = x + jnp.sqrt(2) * layers(x)
        x_final = hk.Linear(self.vocab_size)(x)
        return jax.nn.softmax(x_final)



def get_hk_model(config):
    assert config.model.type == 'transformer'
    
    text_model_fn = lambda *args: TransformerDiffusion(
        config.model.embed_dim,
        config.model.num_layers,
        config.model.d_model,
        config.model.time_dim,
        config.model.num_heads,
        config.tokenizer.vocab_size,
    )(*args)
    model = hk.without_apply_rng(hk.transform(text_model_fn))
    return  model


def get_and_init_model(config, key):
    model = get_hk_model(config)
    key, subkey = jax.random.split(key)    
    text_size = config.seq_len
    vocab_size = config.tokenizer.vocab_size
    params = model.init(
        subkey,
        jnp.zeros([1, text_size, vocab_size], dtype=jnp.int32),
        jnp.zeros([1,], dtype=jnp.int32)
    )
    return params, model.apply
