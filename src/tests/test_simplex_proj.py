import chex
import jax.numpy as jnp
from pathlib import Path
import sys

src_dir = Path(__file__).parent.parent
sys.path.append(src_dir)
print(sys)
from ssm.simplex_proj import _vector_simplex_proj, vector_simplex_proj


def test_vector_simplex_proj():
    x0_single = jnp.array([-1, 1, 0, -1, 0, 2/3])
    simplex_single = jnp.array([0., 2/3, 0., 0., 0., 1/3])
    chex.assert_tree_all_close(
        _vector_simplex_proj(x0_single),
        simplex_single
    )

    x0_batched = jnp.array([[.19, .61], [.61, .65], [.8, .05]])
    simplex_batched = jnp.array([[0.29, 0.71],[0.48, 0.52], [0.875, 0.125]])
    chex.assert_tree_all_close(
        vector_simplex_proj(x0_batched),
        simplex_batched
    )
