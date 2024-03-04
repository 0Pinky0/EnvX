from typing import Tuple

import jax
from jax import lax
import jax.numpy as jnp

rng = jax.random.PRNGKey(0)
x1 = jax.random.uniform(
    key=rng, minval=0, maxval=10
).round().astype(jnp.int32)
_, rng = jax.random.split(rng)
y1 = jax.random.uniform(
    key=rng, minval=0, maxval=10
).round().astype(jnp.int32)
_, rng = jax.random.split(rng)
x2 = jax.random.uniform(
    key=rng, minval=0, maxval=10
).round().astype(jnp.int32)
_, rng = jax.random.split(rng)
y2 = jax.random.uniform(
    key=rng, minval=0, maxval=10
).round().astype(jnp.int32)
_, rng = jax.random.split(rng)
print(f'({x1}, {y1}) -> ({x2}, {y2})')


def bresenham(x1, y1, x2, y2):
    dx = jnp.abs(x2 - x1)
    dy = jnp.abs(y2 - y1)

    x1, y1, x2, y2 = lax.select(
        dy > dx,
        jnp.array([y1, x1, y2, x2]),
        jnp.array([x1, y1, x2, y2]),
    )
    x1, y1, x2, y2 = lax.select(
        x1 > x2,
        jnp.array([x2, y2, x1, y1]),
        jnp.array([x1, y1, x2, y2]),
    )
    # print(f'({x1}, {y1}) -> ({x2}, {y2})')

    dx = jnp.abs(x2 - x1)
    dy = jnp.abs(y2 - y1)
    error = dx // 2
    y = y1
    ystep = lax.select(y1 < y2, 1, -1)

    def bresenham_body(x, val: Tuple[int, float, int, int]):
        y, error, _, _ = val
        x_, y_ = lax.select(dy > dx, jnp.array([y, x]), jnp.array([x, y]))
        print(x_, y_)
        error -= dy
        y += lax.select(error < 0, ystep, 0)
        error += lax.select(error < 0, dx, 0)
        return y, error, x_, y_

    _, _, x_, y_ = lax.fori_loop(x1, x2 + 1, bresenham_body, (y, error, 0, 0))
    return x_, y_
    # points = []
    # for x in range(x1, x2 + 1):
    #     coord = (y, x) if slope else (x, y)
    #     points.append(coord)
    #     error -= dy
    #     if error < 0:
    #         y += ystep
    #         error += dx


if __name__ == '__main__':
    x, y = bresenham(x1, y1, x2, y2)
    print(x, y)
