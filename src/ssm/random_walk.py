import chex
import jax
import jax.numpy as jnp


dim = 4
n = 10_0000

def cube_to_simplex_project(x: TensorType['batch', -1]):
    assert torch.all(x >= 0) and torch.all(x <= 1)
    dist_to_simplex = (1 - x.sum(dim=1)).tolist()
    # black magic
    # more seriously, example, goes from tensor with shape (batch, dim) such as
    # the 3x2 tensor [[.19, .61], [.61, .65], [.8, .05]] to  (batch, dim, dim) 3x2x2
    # [[[0.1900, 0.6100],
    #   [0.1900, 0.6100]],
    #  [[0.6100, 0.6500],
    #   [0.6100, 0.6500]],
    #  [[0.8000, 0.0500],
    #   [0.8000, 0.0500]]]
    many_xs = torch.tile(x[:, None], (1, x.shape[1], 1))
    dist_diags = torch.cat([torch.eye(x.shape[-1])[None] * d for d in dist_to_simplex])
    return torch.mean(dist_diags + many_xs, dim=1)


def walk(starting_point, n: int):
    """
    """
    dim = starting_point.shape[0]
    sigma  = dim
    increments = torch.normal(
        mean=torch.zeros((dim * n,)),
        std=torch.ones((dim * n,)) * (sigma / math.sqrt(dim))
    ).reshape((n, dim))
    x = starting_point + torch.cumsum(increments, dim=1) # Positions
    y = x % 2
    y = torch.where(x > 2, y, y  - 2)
    y, _ = torch.sort(y, dim=0)
    return torch.diff(y, dim=0)


def simplex_projection(x):
    v = x[0]
    rho = v - 1
    v = []
    for i in range(1, v.shape[0]):
        vi = x[i]
        if vi > rho:
            rho += (vi - rho)/(v.abs() + 1)
            if rho > (vi - 1):
                

