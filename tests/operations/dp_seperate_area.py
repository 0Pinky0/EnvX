import jax
from jax import lax
import jax.numpy as jnp
from jax.core import collections

a = jnp.array([
    [0, 0, 1, 0],
    [0, 1, 1, 0],
    [0, 0, 0, 0],
    [0, 1, 0, 0],
])


@jax.jit
def numIslands(grid: jax.Array) -> int:
    nr, nc = grid.shape

    def true_func(val: tuple[int, jax.Array]):
        num_islands, grid = val
        num_islands += 1
        # grid[r][c] = 0
        grid = grid.at[r, c].set(0)
        neighbors = collections.deque([(r, c)])
        while neighbors:
            row, col = neighbors.popleft()
            for x, y in [(row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)]:
                neighbors, grid = lax.cond(
                    0 <= x < nr and 0 <= y < nc and grid[x][y] == 1,
                    body,
                    lambda val: val,
                    (neighbors, grid)
                )
        return num_islands, grid

    def body(val):
        neighbors, grid = val
        return neighbors, grid

    num_islands = 0
    for r in range(nr):
        for c in range(nc):
            num_islands, grid = lax.cond(grid[r][c] == 1, true_func, lambda val: val, (num_islands, grid))

    return num_islands


print(numIslands(a))
