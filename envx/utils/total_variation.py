import jax
import jax.numpy as jnp


@jax.jit
def total_variation(images: jax.Array):
    """https://github.com/tensorflow/tensorflow/blob/v2.7.0/tensorflow/python/ops/image_ops_impl.py#L3213-L3282"""
    pixel_dif1 = images[1:, :] - images[:-1, :]
    pixel_dif2 = images[:, 1:] - images[:, :-1]
    tot_var = jnp.abs(pixel_dif1).sum() + jnp.abs(pixel_dif2).sum()
    return tot_var
