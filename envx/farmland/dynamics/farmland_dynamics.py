from __future__ import annotations

import jax
import jax.numpy as jnp
from gymnasium.experimental.functional import ActType, StateType
from jax.random import PRNGKey

from envx.base.env_dynamics_base import EnvDynamicsBase
from envx.farmland.dynamics.farmland_env_state import FarmlandEnvState


def fell_off(player_position):
    """Checks to see if the player_position means the player has fallen of the cliff."""
    return (
            (player_position[0] == 3)
            * (player_position[1] >= 1)
            * (player_position[1] <= 10)
    )


class FarmlandDynamics(
    EnvDynamicsBase[jax.Array, jax.Array, int, float, bool]
):

    def transition(self, state: FarmlandEnvState, action: int | jax.Array, key: PRNGKey):
        """The Cliffwalking environment's state transition function."""
        new_position = state.player_position

        # where is the agent trying to go?
        new_position = jnp.array(
            [
                new_position[0] + (1 * (action == 2)) + (-1 * (action == 0)),
                new_position[1] + (1 * (action == 1)) + (-1 * (action == 3)),
            ]
        )

        # prevent out of bounds
        new_position = jnp.array(
            [
                jnp.maximum(jnp.minimum(new_position[0], 3), 0),
                jnp.maximum(jnp.minimum(new_position[1], 11), 0),
            ]
        )

        # if we fell off, we have to start over from scratch from (3,0)
        fallen = fell_off(new_position)
        new_position = jnp.array(
            [
                new_position[0] * (1 - fallen) + 3 * fallen,
                new_position[1] * (1 - fallen),
            ]
        )
        new_state = FarmlandEnvState(
            player_position=new_position.reshape((2,)),
            last_action=action[0],
            fallen=fallen,
        )

        return new_state

    def initial(self, rng: PRNGKey) -> FarmlandEnvState:
        """Cliffwalking initial observation function."""
        player_position = jnp.array([3, 0])

        state = FarmlandEnvState(player_position=player_position, last_action=-1, fallen=False)
        return state

    def observation(self, state: FarmlandEnvState) -> int:
        """Cliffwalking observation."""
        return jnp.array(
            state.player_position[0] * 12 + state.player_position[1]
        ).reshape((1,))

    def terminal(self, state: FarmlandEnvState) -> jax.Array:
        """Determines if a particular Cliffwalking observation is terminal."""
        return jnp.array_equal(state.player_position, jnp.array([3, 11]))

    def reward(
            self, state: FarmlandEnvState, action: ActType, next_state: StateType
    ) -> jax.Array:
        """Calculates reward from a state."""
        state = next_state
        reward = -1 + (-99 * state.fallen[0])
        return jax.lax.convert_element_type(reward, jnp.float32)
