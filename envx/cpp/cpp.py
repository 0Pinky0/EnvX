"""Implementation of a Jax-accelerated cartpole environment."""
from __future__ import annotations

from typing import Any, Tuple, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax.random import PRNGKey

import gymnasium as gym
from gymnasium.error import DependencyNotInstalled
from gymnasium.experimental.functional import ActType, FuncEnv, StateType
from gymnasium.experimental.functional_jax_env import (
    FunctionalJaxEnv,
    FunctionalJaxVectorEnv,
)
from gymnasium.utils import EzPickle

RenderStateType = Tuple["pygame.Surface", "pygame.time.Clock"]  # type: ignore  # noqa: F821


class EnvState(NamedTuple):
    position: jnp.array
    angle: jnp.array
    # Channels: [Frontier(unseen), Obstacles, Farmland, Trajectory]
    map_frontier: jnp.array
    map_obstacle: jnp.array
    map_farmland: jnp.array
    map_trajectory: jnp.array
    last_action: int
    fallen: bool

class EnvConfig(NamedTuple):
    time_slice: float
    map_size: jnp.array


class CppFunctional(
    FuncEnv[EnvState, jax.Array, jax.Array, float, bool, RenderStateType]
):
    time_slice = 0.02

    gravity = 9.8
    masscart = 1.0
    masspole = 0.1
    total_mass = masspole + masscart
    length = 0.5
    polemass_length = masspole + length
    force_mag = 10.0
    tau = 0.02
    theta_threshold_radians = 12 * 2 * np.pi / 360
    x_threshold = 2.4
    x_init = 0.05

    screen_width = 600
    screen_height = 600

    action_space = gym.spaces.Box(
        low=np.array([-4, -1800]),
        high=np.array([12, 1800]),
        shape=(2,),
        dtype=np.float32
    )  # 4 directions
    observation_space = gym.spaces.Box(
        low=0.,
        high=1.,
        shape=(4, 80, 80),
        dtype=np.float32
    ),  # Channels: [Frontier(unseen), Obstacles, Farmland, Trajectory]

    def initial(self, rng: PRNGKey):
        """Initial state generation."""
        return jax.random.uniform(
            key=rng, minval=-self.x_init, maxval=self.x_init, shape=(4,)
        )

    def transition(
            self, state: jax.Array, action: int | jax.Array, rng: None = None
    ) -> StateType:
        """Cartpole transition."""
        x, x_dot, theta, theta_dot = state
        force = jnp.sign(action - 0.5) * self.force_mag
        costheta = jnp.cos(theta)
        sintheta = jnp.sin(theta)

        # For the interested reader:
        # https://coneural.org/florian/papers/05_cart_pole.pdf
        temp = (
                       force + self.polemass_length * theta_dot ** 2 * sintheta
               ) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
                self.length * (4.0 / 3.0 - self.masspole * costheta ** 2 / self.total_mass)
        )
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        x = x + self.tau * x_dot
        x_dot = x_dot + self.tau * xacc
        theta = theta + self.tau * theta_dot
        theta_dot = theta_dot + self.tau * thetaacc

        new_state = jnp.array((x, x_dot, theta, theta_dot), dtype=jnp.float32)

        return new_state

    def observation(self, state: jax.Array) -> jax.Array:
        """Cartpole observation."""
        return state

    def terminal(self, state: jax.Array) -> jax.Array:
        """Checks if the state is terminal."""
        x, _, theta, _ = state

        terminated = (
                (x < -self.x_threshold)
                | (x > self.x_threshold)
                | (theta < -self.theta_threshold_radians)
                | (theta > self.theta_threshold_radians)
        )

        return terminated

    def reward(
            self, state: StateType, action: ActType, next_state: StateType
    ) -> jax.Array:
        """Computes the reward for the state transition using the action."""
        x, _, theta, _ = state

        terminated = (
                (x < -self.x_threshold)
                | (x > self.x_threshold)
                | (theta < -self.theta_threshold_radians)
                | (theta > self.theta_threshold_radians)
        )

        reward = jax.lax.cond(terminated, lambda: 0.0, lambda: 1.0)
        return reward

    def render_image(
            self,
            state: StateType,
            render_state: RenderStateType,
    ) -> tuple[RenderStateType, np.ndarray]:
        """Renders an image of the state using the render state."""
        try:
            import pygame
            from pygame import gfxdraw
        except ImportError as e:
            raise DependencyNotInstalled(
                "pygame is not installed, run `pip install gymnasium[classic-control]`"
            ) from e
        screen, clock = render_state

        world_width = self.x_threshold * 2
        scale = self.screen_width / world_width
        polewidth = 10.0
        polelen = scale * (2 * self.length)
        cartwidth = 50.0
        cartheight = 30.0

        x = state

        surf = pygame.Surface((self.screen_width, self.screen_height))
        surf.fill((255, 255, 255))

        l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
        axleoffset = cartheight / 4.0
        cartx = x[0] * scale + self.screen_width / 2.0  # MIDDLE OF CART
        carty = 100  # TOP OF CART
        cart_coords = [(l, b), (l, t), (r, t), (r, b)]
        cart_coords = [(c[0] + cartx, c[1] + carty) for c in cart_coords]
        gfxdraw.aapolygon(surf, cart_coords, (0, 0, 0))
        gfxdraw.filled_polygon(surf, cart_coords, (0, 0, 0))

        l, r, t, b = (
            -polewidth / 2,
            polewidth / 2,
            polelen - polewidth / 2,
            -polewidth / 2,
        )

        pole_coords = []
        for coord in [(l, b), (l, t), (r, t), (r, b)]:
            coord = pygame.math.Vector2(coord).rotate_rad(-x[2])
            coord = (coord[0] + cartx, coord[1] + carty + axleoffset)
            pole_coords.append(coord)
        gfxdraw.aapolygon(surf, pole_coords, (202, 152, 101))
        gfxdraw.filled_polygon(surf, pole_coords, (202, 152, 101))

        gfxdraw.aacircle(
            surf,
            int(cartx),
            int(carty + axleoffset),
            int(polewidth / 2),
            (129, 132, 203),
        )
        gfxdraw.filled_circle(
            surf,
            int(cartx),
            int(carty + axleoffset),
            int(polewidth / 2),
            (129, 132, 203),
        )

        gfxdraw.hline(surf, 0, self.screen_width, carty, (0, 0, 0))

        surf = pygame.transform.flip(surf, False, True)
        screen.blit(surf, (0, 0))

        return (screen, clock), np.transpose(
            np.array(pygame.surfarray.pixels3d(screen)), axes=(1, 0, 2)
        )

    def render_init(
            self, screen_width: int = 600, screen_height: int = 400
    ) -> RenderStateType:
        """Initialises the render state for a screen width and height."""
        try:
            import pygame
        except ImportError as e:
            raise DependencyNotInstalled(
                "pygame is not installed, run `pip install gymnasium[classic-control]`"
            ) from e

        pygame.init()
        screen = pygame.Surface((screen_width, screen_height))
        clock = pygame.time.Clock()

        return screen, clock

    def render_close(self, render_state: RenderStateType) -> None:
        """Closes the render state."""
        try:
            import pygame
        except ImportError as e:
            raise DependencyNotInstalled(
                "pygame is not installed, run `pip install gymnasium[classic-control]`"
            ) from e
        pygame.display.quit()
        pygame.quit()


class CppJaxEnv(FunctionalJaxEnv, EzPickle):
    """Jax-based implementation of the CartPole environment."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 50}

    def __init__(self, render_mode: str | None = None, **kwargs: Any):
        """Constructor for the CartPole where the kwargs are applied to the functional environment."""
        EzPickle.__init__(self, render_mode=render_mode, **kwargs)

        env = CppFunctional(**kwargs)
        env.transform(jax.jit)

        FunctionalJaxEnv.__init__(
            self,
            env,
            metadata=self.metadata,
            render_mode=render_mode,
        )


class CppJaxVectorEnv(FunctionalJaxVectorEnv, EzPickle):
    """Jax-based implementation of the vectorized CartPole environment."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 50}

    def __init__(
            self,
            num_envs: int,
            render_mode: str | None = None,
            max_episode_steps: int = 200,
            **kwargs: Any,
    ):
        """Constructor for the vectorized CartPole where the kwargs are applied to the functional environment."""
        EzPickle.__init__(
            self,
            num_envs=num_envs,
            render_mode=render_mode,
            max_episode_steps=max_episode_steps,
            **kwargs,
        )

        env = CppFunctional(**kwargs)
        env.transform(jax.jit)

        FunctionalJaxVectorEnv.__init__(
            self,
            func_env=env,
            num_envs=num_envs,
            metadata=self.metadata,
            render_mode=render_mode,
            max_episode_steps=max_episode_steps,
        )
