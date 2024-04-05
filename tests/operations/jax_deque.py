import jax
import jax.numpy as jnp
import chex

from flax import struct


@struct.dataclass
class CircularBufferState:
    memory: chex.Array
    index: int
    n_elements: int


def circular_buffer_reset(
        capacity: int, dummy: chex.Array
) -> CircularBufferState:
    buffer_state = CircularBufferState(
        memory=jnp.zeros((capacity, *dummy.shape)),
        index=0,
        n_elements=0
    )
    return buffer_state


@jax.jit
def circular_buffer_push(
        state: CircularBufferState, element: chex.Array
) -> CircularBufferState:
    n_state = CircularBufferState(
        memory=state.memory.at[state.index, :].set(element),
        index=(state.index + 1) % (state.memory.shape[0]),
        n_elements=jnp.maximum(state.index + 1, state.n_elements)
    )
    return n_state
