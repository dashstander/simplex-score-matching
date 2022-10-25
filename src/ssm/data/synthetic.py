
from random import choice, choices
import jax.numpy as jnp


class BiGramSampler:
    vocab = ('a', 'b', 'c')
    
    probs = {
        'a': (1./3, 1./3, 1.3),
        'b': (0.1, 0.4, 0.5),
        'c': (0.6, 0.2, 0.2)
    }

    vectors = {
        'a': jnp.array([1., 0., 0.]),
        'b': jnp.array([0., 1., 0.]),
        'c': jnp.array([0., 0., 1.])
    }
    
    def __init__(self, seq_len: int):
        self.seq_len = seq_len
        
    def _sample(self):
        output = [choice(self.vocab)]
        for _ in range(self.seq_len - 1):
            output += choices(self.vocab, weights=self.probs[output[-1]])
        return ''.join(output)
        
    def sample(self, n: int):
        return [self._sample() for _ in range(n)]

    def to_jax(self, s):
        return jnp.stack([self.vectors[char] for char in s])

    def make_batch(self, n: int):
        tensors = [
            self.to_jax(self._sample()) for _ in range(n)
        ]
        return jnp.stack(tensors)
