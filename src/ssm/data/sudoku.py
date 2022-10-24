
from concurrent.futures import ProcessPoolExecutor
import jax
import jax.numpy as jnp
from sudoku import Sudoku
import sys


class FlatSudoku:
    
    def __init__(self, puzzle):
        self.width = puzzle.width
        self.size = self.width ** 2
        self.puzzle = puzzle
        self.solution = puzzle.solve()
        self.solution_tensor = self.to_tensor()

    def to_tensor(self):
        return jax.nn.one_hot(
            jnp.array(self.solution.board).ravel(),
            num_classes=self.size
        ).astype(jnp.float32)

    @classmethod
    def make_puzzle(cls, seed=None):
        return cls(Sudoku(width=3, height=3, seed=seed))


def generate_masks(rng, batch_size, min_given=0.1, max_given=0.8):
    num_given_key, pos_key = jax.random.split(rng)
    given = jax.random.uniform(
        num_given_key,
        (batch_size,),
        minval=min_given,
        maxval=max_given
    )
    return jax.random.bernoulli(pos_key, p=given, shape=(batch_size, 81))


def make_puzzles(rng, batch_size):
    seeds = jax.random.randint(rng, (batch_size,), 0, sys.maxsize).tolist()
    def _puzzle(s):
        return FlatSudoku.make_puzzle(s).solution_tensor
    with ProcessPoolExecutor(max_workers=64) as exec:
        puzzles = [p for p in exec.map(_puzzle, seeds)]
    return jnp.stack(puzzles)

    
def make_batch(rng, batch_size):
    puzzle_key, mask_key = jax.random.split(rng)
    puzzles = make_puzzles(puzzle_key, batch_size)
    masks = generate_masks(mask_key, batch_size)
    return puzzles, masks
