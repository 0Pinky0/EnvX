import jax
import jax.numpy as jnp

from functools import partial


# @partial(jax.jit, static_argnums=(1, 2))
# def update_map(map: jax.Array, x: int, y: int, r: int) -> jax.Array:
#     leftmost = jnp.maximum(x - r, 0)
#     rightmost = jnp.minimum(x + r, 600)
#     upmost = jnp.maximum(y - r, 0)
#     bottommost = jnp.minimum(y + r, 600)
#     jax.lax.fori_loop(upmost, bottommost)

# l = 9
# x = jnp.array([4])
# y = jnp.array([4])
# # a = jnp.arange(0, 600)
# a = jax.lax.broadcast(jnp.arange(0, l), sizes=[l])
# print(a.dtype)
# print(a.shape)
# b = jax.lax.broadcast(jnp.arange(0, l), sizes=[l]).swapaxes(0, 1)
# print(b.shape)
# c = (a - x) * (a - x) + (b - y) * (b - y)
# print(c)
# d = jnp.where(c <= 16, 0, jnp.ones_like(c))
# print(d)

# paint = jnp.ones([3, 3, 3], dtype=jnp.uint8) * 255
# mask = jnp.ones([3, 3, 3], dtype=jnp.bool_)
# mask = mask.at[1, 1].set(0)
# paint = jnp.where(mask == 0, jnp.array([0, 255, 0]), paint)
# print(paint)

a = jnp.arange(0, 16).reshape(4, 4)
print(a)
print(a.repeat(4, axis=0).repeat(4, axis=1))

# a = jax.lax.dynamic_slice(
#     jnp.ones([5, 5]),
#     start_indices=(3, 3),
#     slice_sizes=(3, 3),
# )
# # print(a)
# b = jax.lax.dynamic_update_slice(
#     jnp.zeros([5, 5]),
#     jnp.ones([2, 2]),
#     start_indices=(3, 3)
# )
# print(b)
# # print(jnp.roll(b, -3, axis=(0)))
# print(jax.lax.dynamic_slice(b, start_indices=(4, 2), slice_sizes=(2, 2)))


