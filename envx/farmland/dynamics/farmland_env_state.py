from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp


class FarmlandEnvState(NamedTuple):
    """A named tuple which contains the full state of the Farmland game."""

    player_position: jnp.array
    last_action: int
    fallen: bool
