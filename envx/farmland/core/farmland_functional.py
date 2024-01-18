"""This module provides a CliffWalking functional environment and Gymnasium environment wrapper CliffWalkingJaxEnv."""

from __future__ import annotations

import jax
import numpy as np
from gymnasium import spaces
from gymnasium.experimental.functional import ActType, FuncEnv, StateType
from jax.random import PRNGKey

from envx.farmland.dynamics.farmland_dynamics import FarmlandDynamics
from envx.farmland.dynamics.farmland_env_state import FarmlandEnvState
from envx.farmland.render.Farmland_render_state import FarmlandRenderState
from envx.farmland.render.Farmland_renderer import FarmlandRenderer


class FarmlandFunctional(
    FuncEnv[jax.Array, jax.Array, int, float, bool, FarmlandRenderState]
):
    """
    Cliff walking involves crossing a gridworld from start to goal while avoiding falling off a cliff.
    """

    action_space = spaces.Box(
        low=np.array([-4, -1800]),
        high=np.array([12, 1800]),
        shape=(2,),
        dtype=np.float32
    )  # 4 directions
    observation_space = spaces.Box(
        low=0.,
        high=1.,
        shape=(4, 80, 80),
        dtype=np.float32
    ),  # Channels: [Frontier(unseen), Obstacles, Farmland, Trajectory]

    metadata = {
        "render_modes": ["rgb_array"],
        "render_fps": 4,
    }

    dynamics = FarmlandDynamics()
    renderer = FarmlandRenderer()

    def transition(self, state: FarmlandEnvState, action: int | jax.Array, key: PRNGKey):
        return self.dynamics.transition(state, action, key)

    def initial(self, rng: PRNGKey) -> FarmlandEnvState:
        return self.dynamics.initial(rng)

    def observation(self, state: FarmlandEnvState) -> int:
        return self.dynamics.observation(state)

    def terminal(self, state: FarmlandEnvState) -> jax.Array:
        return self.dynamics.terminal(state)

    def reward(
            self, state: FarmlandEnvState, action: ActType, next_state: StateType
    ) -> jax.Array:
        return self.dynamics.reward(state, action, next_state)

    def render_init(
            self, screen_width: int = 600, screen_height: int = 500
    ) -> FarmlandRenderState:
        return self.renderer.render_init(screen_width, screen_height)

    def render_image(
            self,
            state: StateType,
            render_state: FarmlandRenderState,
    ) -> tuple[FarmlandRenderState, np.ndarray]:
        return self.renderer.render_image(state, render_state)

    def render_close(self, render_state: FarmlandRenderState) -> None:
        self.renderer.render_close(render_state)
