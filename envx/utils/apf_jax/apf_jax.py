# -*- coding: utf-8 -*-

__all__ = ["apf"]

from functools import partial

import numpy as np
from jax import core, dtypes, lax
from jax import numpy as jnp
from jax.core import ShapedArray
from jax.interpreters import ad, batching, mlir, xla
from jax.lib import xla_client
from jaxlib.hlo_helpers import custom_call

# Register the CPU XLA custom calls
import cpu_apf

for _name, _value in cpu_apf.registrations().items():
    xla_client.register_custom_call_target(_name, _value, platform="cpu")


# This function exposes the primitive to user code and this is the only
# public-facing function in this module


def apf(map_weed):
    return _kepler_prim.bind(map_weed)


# *********************************
# *  SUPPORT FOR JIT COMPILATION  *
# *********************************

# For JIT compilation we need a function to evaluate the shape and dtype of the
# outputs of our op for some given inputs
def _apf_abstract(map_weed):
    shape = map_weed.shape
    # dtype = dtypes.canonicalize_dtype(map_weed.dtype)
    return ShapedArray(shape, jnp.float32)


# We also need a lowering rule to provide an MLIR "lowering" of out primitive.
# This provides a mechanism for exposing our custom C++ and/or CUDA interfaces
# to the JAX XLA backend. We're wrapping two translation rules into one here:
# one for the CPU and one for the GPU
def _apf_lowering(ctx, map_weed, platform="cpu"):
    # Extract the numpy type of the inputs
    map_weed_aval = ctx.avals_in[0]
    np_dtype = np.dtype(map_weed_aval.dtype)

    # The inputs and outputs all have the same shape and memory layout
    # so let's predefine this specification
    # out_dummy = jnp.zeros_like(map_weed, dtype=jnp.float32)
    # dtype = mlir.ir.RankedTensorType(map_weed.type)
    dtype = mlir.ir.RankedTensorType.get(map_weed_aval.shape, mlir.ir.F32Type.get())
    dims = dtype.shape
    assert len(dims) == 2
    layout = tuple(range(len(dims) - 1, -1, -1))

    # The total size of the input is the product across dimensions
    # size = np.prod(dims).astype(np.int64)
    size_dim0 = dims[0]
    size_dim1 = dims[1]

    # We dispatch a different call depending on the dtype
    if np_dtype == np.bool_:
        op_name = platform + "_apf_bool"
    elif np_dtype == np.float32:
        op_name = platform + "_apf_f32"
    elif np_dtype == np.float64:
        op_name = platform + "_apf_f64"
    else:
        raise NotImplementedError(f"Unsupported dtype {np_dtype}")

    # On the CPU, we pass the size of the data as the first input
    # argument
    return custom_call(
        op_name,
        # Output types
        result_types=[dtype],
        # The inputs:
        operands=[mlir.ir_constant(size_dim0), mlir.ir_constant(size_dim1), map_weed],
        # Layout specification:
        operand_layouts=[(), (), layout],
        result_layouts=[layout]
    ).results


# ************************************
# *  SUPPORT FOR BATCHING WITH VMAP  *
# ************************************

# Our op already supports arbitrary dimensions so the batching rule is quite
# simple. The jax.lax.linalg module includes some example of more complicated
# batching rules if you need such a thing.
def _kepler_batch(args, axes):
    assert axes[0] == axes[1]
    return apf(*args), axes


# *********************************************
# *  BOILERPLATE TO REGISTER THE OP WITH JAX  *
# *********************************************
_kepler_prim = core.Primitive("apf")
_kepler_prim.multiple_results = False
_kepler_prim.def_impl(partial(xla.apply_primitive, _kepler_prim))
_kepler_prim.def_abstract_eval(_apf_abstract)

# Connect the XLA translation rules for JIT compilation
for platform in ["cpu"]:  # , "gpu"]:
    mlir.register_lowering(
        _kepler_prim,
        partial(_apf_lowering, platform=platform),
        platform=platform)

# Connect the JVP and batching rules
batching.primitive_batchers[_kepler_prim] = _kepler_batch
