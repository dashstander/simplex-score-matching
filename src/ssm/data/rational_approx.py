from dataclasses import dataclass
from fractions import Fraction
from random import choice, choices
import numpy as np


def int_to_binary(x, width=14):
    return np.array(list(np.binary_repr(x, width=width)), dtype=np.uint8)


@dataclass
class RationalApprox:
    target_real: str
    frac: Fraction
    dtype = np.uint32

    @property
    def numerator(self):
        return self.frac.numerator

    @property
    def denominator(self):
        return self.frac.denominator

    def approximation(self):
        num = self.frac.numerator * 1.
        denom = self.frac.denominator
        return num / denom

    def to_numpy(self):
        num = int_to_binary(self.numerator)
        denom = int_to_binary(self.denominator)
        return np.stack((num, denom))

    def for_batch(self):
        return self.target_real, self.to_numpy()


def rand_frac(decimal_places=15, max_denom=1024):
    # TODO: Refactor this to be in terms of numpy or jax rng for proper reproducibility
    digits = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    leading_digit = choices(digits, weights=[4, 1, 1, 1, 1, 1, 1, 1, 1, 1])[0]
    remaining = ''.join(choices(digits, k=decimal_places))
    number = f'{leading_digit}.{remaining}'
    return RationalApprox(number, Fraction(number).limit_denominator(max_denom))


def make_batch(batch_size=128):
    denominators = [4, 8, 16, 32, 64, 128, 256, 512, 1024]
    numbers = [rand_frac(max_denom=choice(denominators)).for_batch() for _ in range(batch_size)]
    dec_strings, fractions = zip(*numbers)
    return np.array(dec_strings, dtype=np.float32), np.stack(fractions, axis=0)
