from typing import Union, Callable
import functools

import jax
import jax.numpy as jnp

from envx.utils.pix import interpolation


def affine_transform(
        image: jax.Array,
        matrix: jax.Array,
        *,
        offset: Union[jax.Array, jax.Array] = 0.,
        order: int = 1,
        mode: str = "nearest",
        cval: float = 0.0,
) -> jax.Array:
    """Applies an affine transformation given by matrix.

    Given an output image pixel index vector o, the pixel value is determined from
    the input image at position jnp.dot(matrix, o) + offset.

    This does 'pull' (or 'backward') resampling, transforming the output space to
    the input to locate data. Affine transformations are often described in the
    'push' (or 'forward') direction, transforming input to output. If you have a
    matrix for the 'push' transformation, use its inverse (jax.numpy.linalg.inv)
    in this function.

    Args:
      image: a JAX array representing an image. Assumes that the image is
        either HWC or CHW.
      matrix: the inverse coordinate transformation matrix, mapping output
        coordinates to input coordinates. If ndim is the number of dimensions of
        input, the given matrix must have one of the following shapes:

        - (ndim, ndim): the linear transformation matrix for each output
          coordinate.
        - (ndim,): assume that the 2-D transformation matrix is diagonal, with the
          diagonal specified by the given value.
        - (ndim + 1, ndim + 1): assume that the transformation is specified using
          homogeneous coordinates [1]. In this case, any value passed to offset is
          ignored.
        - (ndim, ndim + 1): as above, but the bottom row of a homogeneous
          transformation matrix is always [0, 0, 0, 1], and may be omitted.

      offset: the offset into the array where the transform is applied. If a
        float, offset is the same for each axis. If an array, offset should
        contain one value for each axis.
      order: the order of the spline interpolation, default is 1. The order has
        to be in the range [0-1]. Note that PIX interpolation will only be used
        for order=1, for other values we use `jax.scipy.ndimage.map_coordinates`.
      mode: the mode parameter determines how the input array is extended beyond
        its boundaries. Default is 'nearest'. Modes 'nearest and 'constant' use
        PIX interpolation, which is very fast on accelerators (especially on
        TPUs). For all other modes, 'wrap', 'mirror' and 'reflect', we rely
        on `jax.scipy.ndimage.map_coordinates`, which however is slow on
        accelerators, so use it with care.
      cval: value to fill past edges of input if mode is 'constant'. Default is
        0.0.

    Returns:
      The input image transformed by the given matrix.

    Example transformations:
      Rotation:

      >>> angle = jnp.pi / 4
      >>> matrix = jnp.array([
      ...    [jnp.cos(rotation), -jnp.sin(rotation), 0],
      ...    [jnp.sin(rotation), jnp.cos(rotation), 0],
      ...    [0, 0, 1],
      ... ])
      >>> result = dm_pix.affine_transform(image=image, matrix=matrix)

      Translation can be expressed through either the matrix itself
      or the offset parameter:

      >>> matrix = jnp.array([
      ...   [1, 0, 0, 25],
      ...   [0, 1, 0, 25],
      ...   [0, 0, 1, 0],
      ... ])
      >>> result = dm_pix.affine_transform(image=image, matrix=matrix)
      >>> # Or with offset:
      >>> matrix = jnp.array([
      ...   [1, 0, 0],
      ...   [0, 1, 0],
      ...   [0, 0, 1],
      ... ])
      >>> offset = jnp.array([25, 25, 0])
      >>> result = dm_pix.affine_transform(
              image=image, matrix=matrix, offset=offset)

      Reflection:

      >>> matrix = jnp.array([
      ...   [-1, 0, 0],
      ...   [0, 1, 0],
      ...   [0, 0, 1],
      ... ])
      >>> result = dm_pix.affine_transform(image=image, matrix=matrix)

      Scale:

      >>> matrix = jnp.array([
      ...   [2, 0, 0],
      ...   [0, 1, 0],
      ...   [0, 0, 1],
      ... ])
      >>> result = dm_pix.affine_transform(image=image, matrix=matrix)

      Shear:

      >>> matrix = jnp.array([
      ...   [1, 0.5, 0],
      ...   [0.5, 1, 0],
      ...   [0, 0, 1],
      ... ])
      >>> result = dm_pix.affine_transform(image=image, matrix=matrix)

      One can also combine different transformations matrices:

      >>> matrix = rotation_matrix.dot(translation_matrix)
    """
    # DO NOT REMOVE - Logging usage.

    # jax.assert_rank(image, 3)
    # jax.assert_rank(matrix, {1, 2})
    # jax.assert_rank(offset, {0, 1})
    assert image.ndim == 3

    if matrix.ndim == 1:
        matrix = jnp.diag(matrix)

    if matrix.shape not in [(3, 3), (4, 4), (3, 4)]:
        error_msg = (
            "Expected matrix shape must be one of (ndim, ndim), (ndim,)"
            "(ndim + 1, ndim + 1) or (ndim, ndim + 1) being ndim the image.ndim. "
            f"The affine matrix provided has shape {matrix.shape}.")
        raise ValueError(error_msg)

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

    interpolate_function = _get_interpolate_function(
        mode=mode,
        order=order,
        cval=cval,
    )
    return interpolate_function(image, coordinates)


def rotate(
        image: jax.Array,
        angle: float,
        *,
        order: int = 1,
        mode: str = "nearest",
        cval: float = 0.0,
) -> jax.Array:
    """Rotates an image around its center using interpolation.

    Args:
      image: a JAX array representing an image. Assumes that the image is
        either HWC or CHW.
      angle: the counter-clockwise rotation angle in units of radians.
      order: the order of the spline interpolation, default is 1. The order has
        to be in the range [0,1]. See `affine_transform` for details.
      mode: the mode parameter determines how the input array is extended beyond
        its boundaries. Default is 'nearest'. See `affine_transform` for details.
      cval: value to fill past edges of input if mode is 'constant'. Default is
        0.0.

    Returns:
      The rotated image.
    """
    # DO NOT REMOVE - Logging usage.

    # Calculate inverse transform matrix assuming clockwise rotation.
    c = jnp.cos(angle)
    s = jnp.sin(angle)
    matrix = jnp.array([[c, s, 0], [-s, c, 0], [0, 0, 1]])

    # Use the offset to place the rotation at the image center.
    image_center = (jnp.asarray(image.shape) - 1.) / 2.
    offset = image_center - matrix @ image_center

    return affine_transform(image, matrix, offset=offset, order=order, mode=mode,
                            cval=cval)


def _get_interpolate_function(
        mode: str,
        order: int,
        cval: float = 0.,
) -> Callable[[jax.Array, jax.Array], jax.Array]:
    """Selects the interpolation function to use based on the given parameters.

    PIX interpolations are preferred given they are faster on accelerators. For
    the cases where such interpolation is not implemented by PIX we really on
    jax.scipy.ndimage.map_coordinates. See specifics below.

    Args:
      mode: the mode parameter determines how the input array is extended beyond
        its boundaries. Modes 'nearest and 'constant' use PIX interpolation, which
        is very fast on accelerators (especially on TPUs). For all other modes,
        'wrap', 'mirror' and 'reflect', we rely on
        `jax.scipy.ndimage.map_coordinates`, which however is slow on
        accelerators, so use it with care.
      order: the order of the spline interpolation. The order has to be in the
        range [0, 1]. Note that PIX interpolation will only be used for order=1,
        for other values we use `jax.scipy.ndimage.map_coordinates`.
      cval: value to fill past edges of input if mode is 'constant'.

    Returns:
      The selected interpolation function.
    """
    if mode == "nearest" and order == 1:
        interpolate_function = interpolation.flat_nd_linear_interpolate
    elif mode == "constant" and order == 1:
        interpolate_function = functools.partial(
            interpolation.flat_nd_linear_interpolate_constant, cval=cval)
    else:
        interpolate_function = functools.partial(
            jax.scipy.ndimage.map_coordinates, mode=mode, order=order, cval=cval)
    return interpolate_function
