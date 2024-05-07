from gymnasium.experimental.wrappers.jax_to_numpy import jax_to_numpy
from jax import lax
from jax import numpy as jnp
import jax
import cv2

r_obs = 5
r_self = 4
self_mask = (
                    (lax.broadcast(jnp.arange(0, 2 * r_self + 1), sizes=[2 * r_self + 1]) - r_self) ** 2
                    + (lax.broadcast(jnp.arange(0, 2 * r_self + 1), sizes=[2 * r_self + 1]).swapaxes(0, 1)
                       - r_self) ** 2
            )
print(self_mask)
self_mask = self_mask <= r_self ** 2
print(self_mask)
self_mask = jax_to_numpy(self_mask.astype(jnp.float32))
cv2.imshow("self_mask", self_mask)
cv2.waitKey(0)

