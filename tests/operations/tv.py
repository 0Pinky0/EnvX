import jax
from jax import lax
import jax.numpy as jnp

from envx.cpp.lawn_mowing.utils import total_variation


a = jnp.ones([5, 5])
b = jnp.zeros([3, 3])

c = lax.dynamic_update_slice(a, b, (4, 4,))
c = 1 - c
print(1 - c)
print(total_variation(1 - c))
cx = c[1:, :] - c[:-1, :]
cy = c[:, 1:] - c[:, :-1]
# cx_ = jnp.stack([])
print(jnp.pad(cx, pad_width=[[0, 1], [0, 0]], mode='constant').shape)
print(cy)
