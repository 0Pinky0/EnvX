import jax
from jax import lax
import jax.numpy as jnp

from functools import partial


a = jnp.ones([5, 5])
b = jnp.zeros([3, 3])

c = lax.dynamic_update_slice(a, b, (4, 4,))
print(c)

