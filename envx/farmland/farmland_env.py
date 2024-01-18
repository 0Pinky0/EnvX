"""This module provides a CliffWalking functional environment and Gymnasium environment wrapper CliffWalkingJaxEnv."""

from __future__ import annotations

import jax
from gymnasium.experimental.functional_jax_env import FunctionalJaxEnv
from gymnasium.utils import EzPickle
from gymnasium.wrappers import HumanRendering

from envx.farmland.core.farmland_functional import FarmlandFunctional


class FarmlandJaxEnv(FunctionalJaxEnv, EzPickle):
    """A Gymnasium Env wrapper for the functional cliffwalking env."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 50}

    def __init__(self, render_mode: str | None = None, **kwargs):
        """Initializes Gym wrapper for cliffwalking functional env."""
        EzPickle.__init__(self, render_mode=render_mode, **kwargs)
        env = FarmlandFunctional(**kwargs)
        env.transform(jax.jit)

        super().__init__(
            env,
            metadata=self.metadata,
            render_mode=render_mode,
        )


if __name__ == "__main__":
    """
    Temporary environment tester function.
    """

    env = HumanRendering(FarmlandJaxEnv(render_mode="rgb_array"))

    obs, info = env.reset()
    print(obs, info)

    terminal = False
    while not terminal:
        action = int(input("Please input an action\n"))
        obs, reward, terminal, truncated, info = env.step(action)
        print(obs, reward, terminal, truncated, info)

    exit()
