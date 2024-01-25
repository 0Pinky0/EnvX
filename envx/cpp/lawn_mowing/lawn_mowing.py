"""Implementation of a Jax-accelerated cartpole environment."""
from __future__ import annotations

from typing import Any, Tuple, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax.random import PRNGKey

import gymnasium as gym
from gymnasium.error import DependencyNotInstalled
from gymnasium.experimental.functional import FuncEnv
from gymnasium.experimental.functional_jax_env import (
    FunctionalJaxEnv,
    FunctionalJaxVectorEnv,
)
from gymnasium.utils import EzPickle


class EnvState(NamedTuple):
    position: jax.Array
    theta: jax.Array
    # Channels: [Frontier(unseen), Obstacles, Farmland, Trajectory]
    map_frontier: jax.Array
    map_obstacle: jax.Array
    map_farmland: jax.Array
    map_trajectory: jax.Array
    map_distance: jax.Array
    crashed: bool
    timestep: int


class EnvConfig(NamedTuple):
    map_size: jnp.array


RenderStateType = Tuple["pygame.Surface", "pygame.time.Clock"]


class LawnMowingFunctional(
    FuncEnv[EnvState, jax.Array, jax.Array, float, bool, RenderStateType]
):
    tau: float = 0.5

    r_self: int = 4
    r_obs: int = 32

    max_timestep: int = 1000

    map_width = 200
    map_height = 200

    screen_width = 600
    screen_height = 600

    action_space = gym.spaces.Box(
        low=np.array([0, -1]),
        high=np.array([7, 1]),
        shape=(2,),
        dtype=np.float32
    )  # 4 directions
    observation_space = gym.spaces.Box(
        low=0.,
        high=1.,
        shape=(2, r_obs, r_obs),
        dtype=np.float32
    )  # Channels: [Frontier(unseen), Obstacles, Farmland, Trajectory]

    def initial(self, rng: PRNGKey):
        """Initial state generation."""
        position = jnp.array([50., 50.])
        x, y = position

        map_frontier = jnp.ones([self.map_width, self.map_height])
        xs = jax.lax.broadcast(jnp.arange(0, self.map_width), sizes=[self.map_height])
        ys = jax.lax.broadcast(jnp.arange(0, self.map_height), sizes=[self.map_width]).swapaxes(0, 1)
        map_distance = (xs - x) * (xs - x) + (ys - y) * (ys - y)
        map_frontier = jnp.where(map_distance <= self.r_self * self.r_self, 0, map_frontier)

        state = EnvState(
            position=position,
            theta=jnp.array([0]),
            map_frontier=map_frontier,
            map_obstacle=jnp.zeros([self.map_width, self.map_height]),
            map_farmland=jnp.zeros([self.map_width, self.map_height]),
            map_trajectory=jnp.zeros([self.map_width, self.map_height]),
            map_distance=map_distance,
            crashed=False,
            timestep=0,
        )
        return state

    def transition(
            self, state: EnvState, action: jax.Array, rng: None = None
    ) -> EnvState:
        """Cartpole transition."""
        (position,
         theta,
         map_frontier,
         map_obstacle,
         map_farmland,
         map_trajectory,
         map_distance,
         crashed,
         timestep) = state
        v_linear, v_angular = action

        # Calculate new pos and angle
        cos_theta = jnp.cos(theta)
        sin_theta = jnp.sin(theta)
        new_position = position + v_linear * jnp.array([cos_theta, sin_theta]).squeeze(axis=1) * self.tau
        new_theta = theta + v_angular * self.tau
        new_theta = (new_theta + jnp.pi) % (2 * jnp.pi) - jnp.pi

        # Update Maps
        x, y = new_position.round().astype(jnp.int32)
        # print(x, y)
        xs = jax.lax.broadcast(jnp.arange(0, self.map_width), sizes=[self.map_height])
        ys = jax.lax.broadcast(jnp.arange(0, self.map_height), sizes=[self.map_width]).swapaxes(0, 1)
        x_delta = xs - x
        y_delta = ys - y
        map_distance = x_delta * x_delta + y_delta * y_delta
        map_frontier = jnp.where(map_distance <= self.r_self * self.r_self, 0, map_frontier)

        # Examine whether outbounds
        x, y = new_position
        crashed = (
                (x < 0)
                | (x > self.map_width)
                | (y < 0)
                | (y > self.map_height)
        )

        # Construct new state
        state = EnvState(
            position=new_position,
            theta=new_theta,
            map_frontier=map_frontier,
            map_obstacle=map_obstacle,
            map_farmland=map_farmland,
            map_trajectory=map_trajectory,
            map_distance=map_distance,
            crashed=crashed,
            timestep=timestep + 1,
        )

        return state

    def observation(self, state: EnvState) -> jax.Array:
        """Cartpole observation."""
        (position,
         theta,
         map_frontier,
         map_obstacle,
         map_farmland,
         map_trajectory,
         map_distance,
         crashed,
         timestep) = state
        x, y = position.round().astype(jnp.int32)

        map_leftmost = jnp.maximum(x - self.r_obs, 0)
        map_upmost = jnp.maximum(y - self.r_obs, 0)

        obs_leftmost = jnp.maximum(self.r_obs - x, 0)
        obs_rightmost = jnp.maximum(self.r_obs + x - self.map_width, 0)
        obs_upmost = jnp.maximum(self.r_obs - y, 0)
        obs_bottommost = jnp.maximum(self.r_obs + y - self.map_height, 0)

        crop_leftmost = jnp.maximum(self.r_obs - x, 0)
        crop_rightmost = jnp.minimum(self.r_obs - x + self.map_width, self.r_obs)
        crop_upmost = jnp.maximum(self.r_obs - y, 0)
        crop_bottommost = jnp.minimum(self.r_obs - y + self.map_height, self.r_obs)

        mask_leftmost = jax.lax.broadcast(
            jnp.arange(0, 2 * self.r_obs),
            sizes=[2 * self.r_obs],
        ) < crop_leftmost
        mask_rightmost = jax.lax.broadcast(
            jnp.arange(0, 2 * self.r_obs),
            sizes=[2 * self.r_obs],
        ) > crop_rightmost
        mask_upmost = jax.lax.broadcast(
            jnp.arange(0, 2 * self.r_obs),
            sizes=[2 * self.r_obs],
        ).swapaxes(0, 1) > crop_upmost
        mask_bottommost = jax.lax.broadcast(
            jnp.arange(0, 2 * self.r_obs),
            sizes=[2 * self.r_obs],
        ).swapaxes(0, 1) > crop_bottommost
        mask = jnp.logical_or(mask_leftmost, mask_rightmost)
        mask = jnp.logical_or(mask, mask_upmost)
        mask = jnp.logical_or(mask, mask_bottommost)

        crop_frontier = jax.lax.dynamic_slice(
            map_frontier,
            start_indices=(map_leftmost, map_upmost),
            slice_sizes=(2 * self.r_obs, 2 * self.r_obs),
        )
        roll_frontier = jnp.roll(crop_frontier, shift=(-obs_leftmost + obs_rightmost, -obs_upmost + obs_bottommost))
        obs_frontier = jnp.where(
            mask,
            1,
            roll_frontier
        )

        crop_obstacle = jax.lax.dynamic_slice(
            map_obstacle,
            start_indices=(map_leftmost, map_upmost),
            slice_sizes=(2 * self.r_obs, 2 * self.r_obs),
        )
        roll_obstacle = jnp.roll(crop_obstacle, shift=(-obs_leftmost + obs_rightmost, -obs_upmost + obs_bottommost))
        obs_obstacle = jnp.where(
            mask,
            0,
            roll_obstacle
        )
        obs = jnp.stack([obs_frontier, obs_obstacle])
        return obs

    def terminal(self, state: EnvState) -> jax.Array:
        """Checks if the state is terminal."""
        (position,
         theta,
         map_frontier,
         map_obstacle,
         map_farmland,
         map_trajectory,
         map_distance,
         crashed,
         timestep) = state

        terminated = jnp.logical_or(crashed, timestep >= self.max_timestep)
        return terminated

    def reward(
            self, state: EnvState, action: jax.Array, next_state: EnvState
    ) -> jax.Array:
        """Computes the reward for the state transition using the action."""
        reward_const = -0.1
        reward_crash = jax.lax.select(next_state.crashed, -20, 0)

        covered_t = self.map_width * self.map_height - state.map_frontier.sum()
        covered_tp1 = self.map_width * self.map_height - next_state.map_frontier.sum()
        reward_coverage = (covered_tp1 - covered_t) / 10

        reward = (
                reward_const
                + reward_crash
                + reward_coverage
        )
        return reward

    def render_image(
            self,
            state: EnvState,
            render_state: RenderStateType,
    ) -> tuple[RenderStateType, np.ndarray]:
        """Renders an image of the state using the render state."""
        # print("Render")
        try:
            import pygame
            from pygame import gfxdraw
        except ImportError as e:
            raise DependencyNotInstalled(
                "pygame is not installed, run `pip install gymnasium[classic-control]`"
            ) from e
        screen, clock = render_state

        x, y = state.position.round().astype(jnp.int32)
        # print(x, y)
        img = jnp.ones([self.map_width, self.map_height, 3], dtype=jnp.uint8) * 255
        img = jnp.where(
            jax.lax.broadcast(state.map_frontier, sizes=[3]).transpose(1, 2, 0) == 0,
            jnp.array([65, 227, 72]),
            img
        )
        img = jnp.where(
            jax.lax.broadcast(state.map_distance, sizes=[3]).transpose(1, 2, 0) <= self.r_self * self.r_self,
            jnp.array([255, 0, 0]),
            img
        )

        img = img.repeat(5, axis=0).repeat(5, axis=1)
        img = np.array(img)

        surf = pygame.surfarray.make_surface(img)
        surf = pygame.transform.scale(surf, size=(self.screen_width, self.screen_height))

        screen.blit(surf, (0, 0))

        return (screen, clock), img

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
        screen = pygame.Surface((self.screen_width, self.screen_height))
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


class LawnMowingJaxEnv(FunctionalJaxEnv, EzPickle):
    """Jax-based implementation of the CartPole environment."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 50}

    def __init__(self, render_mode: str | None = None, **kwargs: Any):
        """Constructor for the CartPole where the kwargs are applied to the functional environment."""
        EzPickle.__init__(self, render_mode=render_mode, **kwargs)

        env = LawnMowingFunctional(**kwargs)
        env.transform(jax.jit)

        FunctionalJaxEnv.__init__(
            self,
            env,
            metadata=self.metadata,
            render_mode=render_mode,
        )


class LawnMowingJaxVectorEnv(FunctionalJaxVectorEnv, EzPickle):
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

        env = LawnMowingFunctional(**kwargs)
        env.transform(jax.jit)

        FunctionalJaxVectorEnv.__init__(
            self,
            func_env=env,
            num_envs=num_envs,
            metadata=self.metadata,
            render_mode=render_mode,
            max_episode_steps=max_episode_steps,
        )
