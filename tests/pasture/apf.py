import jax
import jax.numpy as jnp
from gymnasium.experimental.wrappers.jax_to_numpy import jax_to_numpy
from jax import lax
import jax.random as jrng
from jax.random import PRNGKey
import numpy as np
import cv2

rng = jrng.PRNGKey(0)
src = jax.random.uniform(shape=(6, 6), key=rng) <= 0.05
src = src.astype(jnp.float32)
dst = src
decay_factor = 0.9

print(src)

src = src[jnp.newaxis, jnp.newaxis, :, :]
one_mask = src != 0
# print(kernel)
horizon = jnp.ceil(1 / (1 - decay_factor)).astype(jnp.int32)
for i in range(horizon):
    kernel = jnp.array([
        [0., decay_factor, 0.],
        [decay_factor, 1., decay_factor],
        [0., decay_factor, 0.],
    ])[jnp.newaxis, jnp.newaxis, :, :]
    src = lax.conv(
        src,
        kernel,
        window_strides=(1, 1),
        padding='SAME'
    )
    src = lax.clamp(0., src, decay_factor)
    dst = jnp.where(
        jnp.logical_not(one_mask),
        src,
        dst,
    )
    one_mask = jnp.logical_or(src != 0, one_mask)
    src = one_mask.astype(jnp.float32)
    decay_factor *= decay_factor
    # print(one_mask)
    # print(src)
print(dst[0, 0].shape)
# dst = lax.clamp(0.0001, dst, 1.0)
jnp.set_printoptions(precision=3)
print(dst)
# kernel = np.ones((3, 3), np.uint8)
# map_pasture = jax_to_numpy(map_pasture)
# dilated1 = cv2.dilate(map_pasture, kernel, iterations=1)
# print(dilated1)
