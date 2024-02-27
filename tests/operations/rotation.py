import jax
import jax.numpy as jnp

from envx.utils.pix.jitted import rotate_nearest

is_hwc = True

a = jnp.array([[
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
]]).astype(jnp.float32)
if is_hwc:
    a = a.transpose(1, 2, 0)
b = rotate_nearest(
    a,
    angle=jnp.pi / 4,
    # mode='constant',
    # cval=1.
)
if is_hwc:
    print(a.transpose(2, 0, 1))
    print(b.transpose(2, 0, 1))
else:
    print(a)
    print(b)
