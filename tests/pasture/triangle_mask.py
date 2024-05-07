from typing import Sequence

from gymnasium.experimental.wrappers.jax_to_numpy import jax_to_numpy
from jax import lax
from jax import numpy as jnp
import jax
import cv2

r_obs = 5
r_self = 4
w_self = 4
# self_mask = (
#                     (lax.broadcast(jnp.arange(0, 2 * r_self + 1), sizes=[2 * r_self + 1]) - r_self) ** 2
#                     + (lax.broadcast(jnp.arange(0, 2 * r_self + 1), sizes=[2 * r_self + 1]).swapaxes(0, 1)
#                        - r_self) ** 2
#             )
# print(self_mask)
# self_mask = self_mask <= r_self ** 2
# print(self_mask)
# self_mask = jax_to_numpy(self_mask.astype(jnp.float32))

s_x = lax.broadcast(jnp.arange(0, 2 * r_self + 1), sizes=[2 * r_self + 1])
s_y = lax.broadcast(jnp.arange(0, 2 * r_self + 1), sizes=[2 * r_self + 1]).swapaxes(0, 1)
print(s_x)
print(s_y)

z_1 = (r_self, r_self)
z_2 = (r_self - w_self, 0)
z_3 = (r_self + w_self, 0)


def to_left(p: Sequence[int], q: Sequence[int], s_x: jax.Array, s_y: jax.Array) -> jax.Array:
    p_x, p_y = p
    q_x, q_y = q
    return (0 <= p_x * q_y - p_y * q_x
            + q_x * s_y - q_y * s_x
            + s_x * p_y - s_y * p_x)


def in_triangle(z_1: Sequence[int], z_2: Sequence[int], z_3: Sequence[int], r: int) -> jax.Array:
    s_x = lax.broadcast(jnp.arange(0, 2 * r + 1), sizes=[2 * r + 1])
    s_y = lax.broadcast(jnp.arange(0, 2 * r + 1), sizes=[2 * r + 1]).swapaxes(0, 1)
    return jnp.logical_and(
        to_left(z_1, z_2, s_x, s_y) == to_left(z_2, z_3, s_x, s_y),
        to_left(z_2, z_3, s_x, s_y) == to_left(z_3, z_1, s_x, s_y),
    )


final_mask = in_triangle(z_1, z_2, z_3, r_self)
print(final_mask)
final_mask = jax_to_numpy(final_mask.astype(jnp.float32))
cv2.imshow("self_mask", final_mask)
cv2.waitKey(0)