import jax
import jax.numpy as jnp
import numpy as np
from sudoku import Sudoku


class FlatSudoku:
    
    def __init__(self, puzzle):
        self.width = puzzle.width
        self.size = self.width ** 2
        self.puzzle = puzzle
        self.solution = puzzle.solve()
        self.solution_tensor = self.to_tensor()

    def to_tensor(self):
        return jax.nn.one_hot(
            jnp.array(self.solution.board).ravel() - 1,
            num_classes=self.size,
            dtype=jnp.int32
        ).astype(jnp.float32)

    @classmethod
    def make_puzzle(cls, seed=None):
        return cls(Sudoku(width=3, height=3, seed=seed))


def generate_masks(rng, batch_size):
    k1, k2 = jax.random.split(rng)
    masks = jax.random.bernoulli(k1, p=0.35, shape=(81, batch_size))
    masks = jnp.transpose(masks)
    cfg_mask = jax.random.bernoulli(k2, shape=(batch_size, 1))
    return masks * cfg_mask


def make_train_loader(config, rng_key):
    key, subkey = jax.random.split(rng_key)
    data_config = config['data']
    batch_size = data_config['batch_size']
    num_train_batches = data_config['num_train_batches']
    num_validation = data_config['num_val_batches'] * batch_size
    data_fp = data_config['data_path']
    data = np.load(data_fp, mmap_mode='r')
    data_size = data['puzzles'].shape[0]
    train_size = data_size - num_validation
   
    for _ in range(num_train_batches):
        key, sk1, sk2 = jax.random.split(key, 3)
        batch_indices = jax.random.choice(sk1, train_size, (batch_size,))
        batch = data['puzzles'][batch_indices] - 1
        masks = generate_masks(sk2, batch_size)
        #givens = batch * masks
        puzzles = jax.nn.one_hot(batch, num_classes=9)
        givens = puzzles * masks[..., None]
        yield puzzles, givens


def make_val_loader(config, rng_key):
    key, subkey = jax.random.split(rng_key)
    data_config = config['data']
    batch_size = data_config['batch_size']
    num_validation = data_config['num_val_batches'] * batch_size
    data_fp = data_config['data_path']
    data = np.load(data_fp, mmap_mode='r')
    data_size = data['puzzles'].shape[0]
    train_size = data_size - num_validation
    indices = np.arange(train_size, data_size)
    splits = data_config['num_val_batches']
    for batch_indices in jnp.array_split(indices, splits):
        if len(batch_indices) != batch_size:
            #print(f'Batch has size {len(batch_indices)} not {batch_size}')
            continue
        key, subkey = jax.random.split(key)
        batch = data['puzzles'][batch_indices] - 1
        masks = data['masks'][batch_indices]
        #givens = batch * masks
        puzzles = jax.nn.one_hot(batch, num_classes=9)
        givens = puzzles * masks[..., None]
        yield puzzles, givens


"""
def get_puzzle(seed):
    return FlatSudoku.make_puzzle(seed).solution_tensor


def make_puzzles(rng, batch_size):
    max_seed = jnp.iinfo(jnp.int32).max
    seeds = jax.random.randint(rng, (batch_size,), 0, max_seed, dtype=jnp.int32).tolist()
    puzzles = [get_puzzle(s) for s in seeds]
    return jnp.stack(puzzles)

    
def make_batch(rng, batch_size):
    puzzle_key, mask_key = jax.random.split(rng)
    puzzles = make_puzzles(puzzle_key, batch_size)
    masks = generate_masks(mask_key, batch_size)
    return puzzles, masks[..., None]
"""
