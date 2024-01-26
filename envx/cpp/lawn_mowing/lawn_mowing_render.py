"""Implementation of a Jax-accelerated cartpole environment."""
from __future__ import annotations

from typing import Any, Tuple, NamedTuple, Dict

import jax
from jax import lax
import jax.numpy as jnp
import numpy as np
from jax.random import PRNGKey

import gymnasium as gym
from gymnasium.error import DependencyNotInstalled
from gymnasium.experimental.functional import FuncEnv

from envx.cpp.lawn_mowing.lawn_mowing import LawnMowingFunctional
from envx.utils.functional_jax_env import (
    FunctionalJaxEnv,
    FunctionalJaxVectorEnv,
)
from gymnasium.experimental.wrappers.jax_to_numpy import jax_to_numpy
from gymnasium.utils import EzPickle

from envx.cpp.lawn_mowing.utils import total_variation


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


class LawnMowingRenderFunctional(
    LawnMowingFunctional
):
    observation_space = gym.spaces.dict.Dict({
        "observations": gym.spaces.Box(
            low=0.,
            high=1.,
            shape=(2, 2 * LawnMowingFunctional.r_obs, 2 * LawnMowingFunctional.r_obs),
            dtype=np.float32
        ),
        "pose": gym.spaces.Box(low=-1., high=1., shape=(2,), dtype=np.float32),
        'pixels': gym.spaces.Box(
            low=0,
            high=255,
            shape=(200, 200, 3),
            dtype=np.uint8
        )
    })  # Channels: [Frontier(unseen), Obstacles, Farmland, Trajectory]

    def observation(self, state: EnvState) -> Dict[str, jax.Array]:
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
        cos_theta = jnp.cos(theta)
        sin_theta = jnp.sin(theta)
        pose = jnp.array([cos_theta, sin_theta]).squeeze(axis=1)
        # Frontier
        map_frontier_aug = jnp.zeros(
            [self.map_height + 2 * self.r_obs, self.map_width + 2 * self.r_obs],
            dtype=jnp.bool_
        )
        map_frontier_aug = lax.dynamic_update_slice(
            map_frontier_aug,
            map_frontier,
            start_indices=(self.r_obs, self.r_obs)
        )
        obs_frontier = lax.dynamic_slice(
            map_frontier_aug,
            start_indices=(y, x),
            slice_sizes=(2 * self.r_obs, 2 * self.r_obs)
        )
        # Obstacle
        map_obstacle_aug = jnp.ones(
            [self.map_height + 2 * self.r_obs, self.map_width + 2 * self.r_obs],
            dtype=jnp.bool_
        )
        map_obstacle_aug = lax.dynamic_update_slice(
            map_obstacle_aug,
            map_obstacle,
            start_indices=(self.r_obs, self.r_obs)
        )
        obs_obstacle = lax.dynamic_slice(
            map_obstacle_aug,
            start_indices=(y, x),
            slice_sizes=(2 * self.r_obs, 2 * self.r_obs)
        )

        obs = jnp.stack([obs_frontier, obs_obstacle], dtype=jnp.float32)

        # Draw
        obs_cols = lax.broadcast(
            jnp.arange(0, self.map_width),
            sizes=[self.map_height]
        )
        obs_rows = lax.broadcast(
            jnp.arange(0, self.map_height),
            sizes=[self.map_width]
        ).swapaxes(0, 1)
        mask_cols_range = jnp.logical_or(
            obs_cols == x - self.r_obs,
            obs_cols == x + self.r_obs
        )
        mask_rows_range = jnp.logical_or(
            obs_rows == y - self.r_obs,
            obs_rows == y + self.r_obs
        )
        mask_cols_condition = jnp.logical_and(
            obs_rows >= y - self.r_obs,
            obs_rows <= y + self.r_obs
        )
        mask_rows_condition = jnp.logical_and(
            obs_cols >= x - self.r_obs,
            obs_cols <= x + self.r_obs,
        )
        mask_cols = jnp.logical_and(
            mask_cols_range,
            mask_cols_condition
        )
        mask_rows = jnp.logical_and(
            mask_rows_range,
            mask_rows_condition,
        )
        mask = jnp.logical_or(
            mask_cols,
            mask_rows
        )
        # TV visualize
        mask_tv_cols = state.map_frontier.astype(jnp.uint8)[1:, :] - state.map_frontier.astype(jnp.uint8)[:-1, :] != 0
        mask_tv_cols = jnp.pad(mask_tv_cols, pad_width=[[0, 1], [0, 0]], mode='constant')
        mask_tv_rows = state.map_frontier.astype(jnp.uint8)[:, 1:] - state.map_frontier.astype(jnp.uint8)[:, :-1] != 0
        mask_tv_rows = jnp.pad(mask_tv_rows, pad_width=[[0, 0], [0, 1]], mode='constant')
        mask_tv = jnp.logical_or(mask_tv_cols, mask_tv_rows)
        # Draw covered area and agent
        img = jnp.ones([self.map_height, self.map_width, 3], dtype=jnp.uint8) * 255
        img = jnp.where(
            lax.broadcast(state.map_frontier, sizes=[3]).transpose(1, 2, 0) == 0,
            jnp.array([65, 227, 72], dtype=jnp.uint8),
            img
        )
        img = jnp.where(
            lax.broadcast(state.map_distance, sizes=[3]).transpose(1, 2, 0) <= self.r_self * self.r_self,
            jnp.array([255, 0, 0], dtype=jnp.uint8),
            img
        )
        img = jnp.where(
            lax.broadcast(mask, sizes=[3]).transpose(1, 2, 0),
            jnp.array([0, 0, 255], dtype=jnp.uint8),
            img
        )
        img = jnp.where(
            lax.broadcast(mask_tv, sizes=[3]).transpose(1, 2, 0),
            jnp.array([255, 38, 255], dtype=jnp.uint8),
            img
        )
        # Scale up the img
        # img = (img
        #        .transpose(2, 0, 1)
        #        .reshape(3, self.map_height, self.map_width // 2, self.map_width // 2)
        #        .mean(axis=-1)
        #        .transpose(0, 2, 1)
        #        .reshape(3, self.map_width // 2, self.map_height // 2, self.map_height // 2)
        #        .mean(axis=-1)
        #        .transpose(2, 1 ,0)
        #        )
        return {
            'observations': obs,
            'pose': pose,
            'pixels': img,
        }


class LawnMowingRenderJaxEnv(FunctionalJaxEnv, EzPickle):
    """Jax-based implementation of the CartPole environment."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 50}

    def __init__(self, render_mode: str | None = None, **kwargs: Any):
        """Constructor for the CartPole where the kwargs are applied to the functional environment."""
        EzPickle.__init__(self, render_mode=render_mode, **kwargs)

        env = LawnMowingRenderFunctional(**kwargs)
        env.transform(jax.jit)

        FunctionalJaxEnv.__init__(
            self,
            env,
            metadata=self.metadata,
            render_mode=render_mode,
        )


class LawnMowingRenderJaxVectorEnv(FunctionalJaxVectorEnv, EzPickle):
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

        env = LawnMowingRenderFunctional(**kwargs)
        env.transform(jax.jit)

        FunctionalJaxVectorEnv.__init__(
            self,
            func_env=env,
            num_envs=num_envs,
            metadata=self.metadata,
            render_mode=render_mode,
            max_episode_steps=max_episode_steps,
        )
