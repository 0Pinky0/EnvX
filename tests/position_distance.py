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

l = 9
x = jnp.array([4])
y = jnp.array([4])
# a = jnp.arange(0, 600)
a = jax.lax.broadcast(jnp.arange(0, l), sizes=[l])
print(a.shape)
b = jax.lax.broadcast(jnp.arange(0, l), sizes=[l]).swapaxes(0, 1)
print(b.shape)
c = (a - x) * (a - x) + (b - y) * (b - y)
print(c)
d = jnp.where(c <= 16, 0, jnp.ones_like(c))
print(d)

