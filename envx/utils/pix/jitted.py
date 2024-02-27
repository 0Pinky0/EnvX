from typing import Union, Callable
import functools

import jax
import jax.numpy as jnp

from envx.utils.pix import interpolation


@jax.jit
def rotate_nearest(
        image: jax.Array,
        angle: float,
) -> jax.Array:
    # DO NOT REMOVE - Logging usage.

    # Calculate inverse transform matrix assuming clockwise rotation.
    c = jnp.cos(angle)
    s = jnp.sin(angle)
    matrix = jnp.array([[c, s, 0], [-s, c, 0], [0, 0, 1]])

    # Use the offset to place the rotation at the image center.
    image_center = (jnp.asarray(image.shape) - 1.) / 2.
    offset = image_center - matrix @ image_center

    meshgrid = jnp.meshgrid(*[jnp.arange(size) for size in image.shape],
                            indexing="ij")
    indices = jnp.concatenate(
        [jnp.expand_dims(x, axis=-1) for x in meshgrid], axis=-1)

    if matrix.shape == (4, 4) or matrix.shape == (3, 4):
        offset = matrix[:image.ndim, image.ndim]
        matrix = matrix[:image.ndim, :image.ndim]

    coordinates = indices @ matrix.T
    coordinates = jnp.moveaxis(coordinates, source=-1, destination=0)

    # Alter coordinates to account for offset.
    offset = jnp.full((3,), fill_value=offset)
    coordinates += jnp.reshape(a=offset, newshape=(*offset.shape, 1, 1, 1))

    interpolate_function = jax.jit(interpolation.flat_nd_linear_interpolate)
    # interpolate_function = interpolation.flat_nd_linear_interpolate
    return interpolate_function(image, coordinates)
