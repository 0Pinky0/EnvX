import jax
import jax.numpy as jnp
from gymnasium.experimental.wrappers.jax_to_numpy import jax_to_numpy
from jax import lax
from envx.utils.apf_jax import apf
# import cpu_apf

import jax.random as jrng
import cv2
import numpy as np

rng = jrng.PRNGKey(0)
a = jax.random.uniform(shape=(6, 6), key=rng) <= 0.1
ori_mask = np.broadcast_to(jax_to_numpy(a), shape=(3, 6, 6)).transpose(1, 2, 0)
ori = np.where(ori_mask, np.array([0., 0., 1.]), np.array([1., 1., 1.]))
# cv2.imshow('ori', ori)
# cv2.imwrite('ori_mask.png', (ori * 255).repeat(50, axis=0).repeat(50, axis=1))

# a = jnp.zeros([6, 6], dtype=jnp.bool_)
print(a)
b = apf(a)
jnp.set_printoptions(precision=2)
print(b)
# print(0.8 ** b)
dst_mask = jax_to_numpy((0.8 ** b))
dst = np.broadcast_to((1 - dst_mask), shape=(3, 6, 6)).transpose(1, 2, 0) * np.array(
        [1.8, 1.8, 0.], dtype=np.float32
    ) + np.array(
        [0., 0., 1.], dtype=np.float32
    )
# cv2.imshow('dst', dst)
# cv2.imwrite('dst_mask.png', (dst * 255).repeat(50, axis=0).repeat(50, axis=1))
# cv2.waitKey(0)

