import jax
from jax import numpy as jnp
from jax import lax
import numpy as np

map_aug = jnp.full(
    [3, 16, 16],
    fill_value=jnp.broadcast_to(jnp.array([0., 0., 1.]), shape=(1, 1, 3)).transpose(2, 0, 1),
    dtype=jnp.float32
)
print(map_aug)

# a = jnp.arange(0, 10)
# a = jnp.broadcast_to(a, shape=(3, 10, 10))
# print(a)
#
# b = lax.reduce_window(a, -jnp.inf, lax.max, (1, 2, 2), (1, 2, 2), padding='VALID')
# print(b)
#
# c = lax.dynamic_slice(
#     b,
#     start_indices=(1, 2, 2),
#     slice_sizes=(3, 2, 2)
# )
# print(c)
