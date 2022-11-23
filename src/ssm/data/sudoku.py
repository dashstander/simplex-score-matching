import jax
import jax.numpy as jnp
import numpy as np
from sudoku import Sudoku

from ssm.utils import split_and_stack


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


def permute_rows_and_blocks(rng, puzzle):
    row_key, block_key = jax.random.split(rng)
    row_inds = jnp.tile(jnp.array([0, 1, 2]), (3, 1))
    row_perms = jax.random.permutation(row_key, row_inds, axis=-1, independent=True)
    row_perms = (row_perms + jnp.arange(0, 9, 3)[..., None]).ravel()
    puzzle = puzzle[row_perms]
    block_perms = jnp.arange(9).reshape((3, 3))[jax.random.permutation(block_key, 3)].ravel()
    return puzzle[block_perms]


def rotate_puzzle(rng, puzzle):
    rot_key, ref_key = jax.random.split(rng)
    fns = [lambda x: jnp.rot90(x, i) for i in range(4)]
    num_rotations = jax.random.randint(rot_key, minval=0, maxval=3, shape=())
    puzzle = jax.lax.switch(num_rotations, fns, puzzle)
    return jax.lax.cond(
        jax.random.bernoulli(ref_key),
        jnp.transpose,
        lambda x: x,
        puzzle
    )


def permute_values(rng, puzzle):
    permutation = jax.random.permutation(rng, 9).astype(jnp.int32)
    return jax.lax.map(lambda x: permutation[x], puzzle)


def generate_permutation(rng, puzzle):
    # The three "block rows" can be freely permuted (1 in S_3)
    # The three "block columns" can be freely permuted (1 in S_3)
    # Each three rows in a block row can be permuted (3 in S_3)
    # Each three columns in a block column can be permuted (3 in S_3)
    # The whole puzzle can be rotated 90, 180, or 270 degress (1 from C_4)
    # The whole puzzle can be reflected about the diagonal (1 reflection)
    # The entire placement of 1s, 2s, etc... can be permuted (1 from S_9)
    # Total: 8 permutations from S_3, one rotation, one reflection,
    # one rotation from S_9
    row_key, col_key, rot_key, s9_key = jax.random.split(rng, 4)
    puzzle = puzzle.reshape((9, 9))
    puzzle = permute_rows_and_blocks(row_key, puzzle)
    # transpose to work on columns, then transpose back
    puzzle = jnp.transpose(
        permute_rows_and_blocks(col_key, jnp.transpose(puzzle))
    )
    puzzle = rotate_puzzle(rot_key, puzzle)
    puzzle = permute_values(s9_key, puzzle.ravel())
    return puzzle


def generate_masks(rng, batch_size):
    k1, k2 = jax.random.split(rng)
    masks = jax.random.bernoulli(k1, p=0.35, shape=(81, batch_size))
    masks = jnp.transpose(masks)
    cfg_mask = jax.random.bernoulli(k2, shape=(batch_size, 1))
    return masks * cfg_mask


def make_train_loader(config, rng):
    data_config = config['data']
    batch_size = data_config['batch_size']
    num_train_batches = data_config['num_train_batches']
    num_validation = data_config['num_val_batches'] * batch_size
    data_fp = data_config['data_path']
    data = np.load(data_fp, mmap_mode='r')
    data_size = data['puzzles'].shape[0]
    train_size = data_size - num_validation
   
    for _ in range(num_train_batches):
        rng, sk1, sk2, sk3 = jax.random.split(rng, 4)
        keys = split_and_stack(sk1, batch_size)
        batch_indices = jax.random.choice(sk2, train_size, (batch_size,))
        batch = data['puzzles'][batch_indices] - 1
        batch = jax.vmap(generate_permutation)(keys, batch)
        masks = generate_masks(sk3, batch_size)
        #givens = batch * masks
        puzzles = jax.nn.one_hot(batch, num_classes=9)
        givens = puzzles * masks[..., None]
        yield puzzles, givens


def make_val_loader(config, rng_key):
    key, subkey = jax.random.split(rng_key)
    data_config = config['data']
    batch_size = data_config['batch_size']
    assert batch_size % 32 == 0
    val_batch_size = batch_size // 32
    num_validation = data_config['num_val_batches'] * batch_size
    data_fp = data_config['data_path']
    data = np.load(data_fp, mmap_mode='r')
    data_size = data['puzzles'].shape[0]
    train_size = data_size - num_validation
    indices = np.arange(train_size, data_size)
    splits = data_config['num_val_batches'] * 32
    all_splits = jnp.array_split(indices, splits)
    for batch_indices in all_splits:
        if len(batch_indices) != val_batch_size:
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
