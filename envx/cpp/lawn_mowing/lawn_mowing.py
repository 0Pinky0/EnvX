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
from envx.utils.functional_jax_env import (
    FunctionalJaxEnv,
    FunctionalJaxVectorEnv,
)
from gymnasium.experimental.wrappers.jax_to_numpy import jax_to_numpy
from gymnasium.utils import EzPickle

from envx.cpp.lawn_mowing import utils
from envx.utils.pix.jitted import rotate_nearest


class EnvState(NamedTuple):
    position: jax.Array
    theta: jax.Array
    # Channels: [Frontier(unseen), Obstacles, Farmland, Trajectory]
    map_frontier: jax.Array
    map_obstacle: jax.Array
    map_farmland: jax.Array
    map_trajectory: jax.Array
    crashed: bool
    timestep: int


RenderStateType = Tuple["pygame.Surface", "pygame.time.Clock"]


class LawnMowingFunctional(
    FuncEnv[EnvState, jax.Array, jax.Array, float, bool, RenderStateType]
):
    tau: float = 0.5

    r_self: int = 4
    r_obs: int = 48
    diag_obs: int = np.ceil(np.sqrt(2) * r_obs).astype(np.int32)

    max_timestep: int = 1000

    map_width = 140
    map_height = 140

    screen_width = 600
    screen_height = 600

    v_max = 7
    w_max = 1
    nvec = [4, 9]

    vision_mask = (
                          (lax.broadcast(jnp.arange(0, map_width), sizes=[map_height]) - map_width // 2) ** 2
                          + (lax.broadcast(jnp.arange(0, map_height), sizes=[map_width]).swapaxes(0, 1)
                             - map_height // 2) ** 2
                  ) <= r_self * r_self

    def __init__(
            self,
            save_pixels: bool = False,
            action_type: str = "continuous",
            rotate_obs: bool = False,
            pbc: bool = False,  # Periodic Boundary Conditions, PBC
            **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.save_pixels = save_pixels
        self.action_type = action_type
        self.rotate_obs = rotate_obs
        self.pbc = pbc
        # Channels: [Frontier(unseen), Obstacles]
        # Future: [Farmland, Trajectory]
        # Define obs space
        obs_dict = {
            "observation": gym.spaces.Box(
                low=0.,
                high=1.,
                shape=(2, 2 * self.r_obs, 2 * self.r_obs),
                dtype=np.float32
            )
        }
        if save_pixels:
            obs_dict["pixels"] = gym.spaces.Box(
                low=0,
                high=255,
                shape=(self.map_height, self.map_width, 3),
                dtype=np.uint8
            )
        if not rotate_obs:
            obs_dict["pose"] = gym.spaces.Box(low=-1., high=1., shape=(2,), dtype=np.float32)
        self.observation_space = gym.spaces.dict.Dict(obs_dict)
        # Define action space
        match action_type:
            case "continuous":
                self.action_space = gym.spaces.Box(
                    low=np.array([0, -self.w_max]),
                    high=np.array([self.v_max, self.w_max]),
                    shape=(2,),
                    dtype=np.float32
                )
            case "discrete":
                self.action_space = gym.spaces.Discrete(
                    n=self.nvec[0] * self.nvec[1]
                )
            case "multi_discrete":
                self.action_space = gym.spaces.MultiDiscrete(
                    nvec=self.nvec
                )
            case _:
                raise ValueError(f"Action type should be continuous, discrete or multi_discrete, got '{action_type}'")
        if pbc:
            if rotate_obs:
                assert 2 * jnp.sqrt(2) * self.r_obs <= min(self.map_width, self.map_height)
            else:
                assert 2 * self.r_obs <= min(self.map_width, self.map_height)

    def initial(self, rng: PRNGKey):
        """Initial state generation."""
        x = jax.random.uniform(
            key=rng, minval=5 * self.r_self, maxval=self.map_height - 5 * self.r_self
        )
        _, rng = jax.random.split(rng)
        y = jax.random.uniform(
            key=rng, minval=5 * self.r_self, maxval=self.map_width - 5 * self.r_self
        )
        position = jnp.stack([x, y])
        _, rng = jax.random.split(rng)
        theta = jax.random.uniform(key=rng, minval=-jnp.pi, maxval=jnp.pi, shape=[1])

        x, y = position
        map_frontier = jnp.ones([self.map_height, self.map_width], dtype=jnp.bool_)
        map_frontier = jnp.where(
            jnp.roll(
                self.vision_mask,
                shift=(y - self.map_height // 2, x - self.map_width // 2),
                axis=(0, 1)
            ),
            False,
            map_frontier,
        )

        state = EnvState(
            position=position,
            theta=theta,
            map_frontier=map_frontier,
            map_obstacle=jnp.zeros([self.map_height, self.map_width], dtype=jnp.bool_),
            map_farmland=jnp.zeros([self.map_height, self.map_width], dtype=jnp.bool_),
            map_trajectory=jnp.zeros([self.map_height, self.map_width], dtype=jnp.bool_),
            crashed=False,
            timestep=0,
        )
        return state

    def get_velocity(self, action) -> Tuple[float, float]:
        match self.action_type:
            case "continuous":
                v_linear, v_angular = action
            case "discrete":
                action_linear = action // self.nvec[1] + 1
                linear_size = self.nvec[0] - 1 + 1
                v_linear = self.v_max * action_linear / linear_size
                action_angular = action % self.nvec[1]
                angular_size = (self.nvec[1] - 1) // 2
                v_angular = self.w_max * (action_angular - 1 - angular_size) / angular_size
            case "multi_discrete":
                linear_size = self.nvec[0] - 1
                v_linear = self.v_max * action[0] / linear_size
                angular_size = (self.nvec[1] - 1) // 2
                v_angular = self.w_max * (action[1] - 1 - angular_size) / angular_size
        return v_linear, v_angular  # noqa

    def transition(
            self, state: EnvState, action: jax.Array, rng: None = None
    ) -> EnvState:
        """Cartpole transition."""
        v_linear, v_angular = self.get_velocity(action)

        # Calculate new pos and angle
        cos_theta = jnp.cos(state.theta)
        sin_theta = jnp.sin(state.theta)
        x, y = state.position + v_linear * jnp.array([cos_theta, sin_theta]).squeeze(axis=1) * self.tau
        if self.pbc:
            x = (x + self.map_width) % (1.0 * self.map_width)
            y = (y + self.map_height) % (1.0 * self.map_height)
        else:
            x = lax.clamp(0., x, float(self.map_width))
            y = lax.clamp(0., y, float(self.map_height))
        new_position = jnp.array([x, y])
        new_theta = state.theta + v_angular * self.tau
        new_theta = (new_theta + jnp.pi) % (2 * jnp.pi) - jnp.pi

        # Update Maps
        x, y = new_position.round().astype(jnp.int32)

        map_frontier = state.map_frontier
        map_frontier = jnp.where(
            jnp.roll(
                self.vision_mask,
                shift=(y - self.map_height // 2, x - self.map_width // 2),
                axis=(0, 1)
            ),
            False,
            map_frontier,
        )

        # Examine whether outbounds
        crashed = False if self.pbc else (
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
            map_obstacle=state.map_obstacle,
            map_farmland=state.map_farmland,
            map_trajectory=state.map_trajectory,
            crashed=crashed,
            timestep=state.timestep + 1,
        )

        return state

    @staticmethod
    @jax.jit
    def crop_obs(map: jax.Array, x: int, y: int, pad_ones: bool = True) -> jax.Array:
        map_aug = jnp.full(
            [LawnMowingFunctional.map_height + 2 * LawnMowingFunctional.r_obs,
             LawnMowingFunctional.map_width + 2 * LawnMowingFunctional.r_obs],
            fill_value=pad_ones,
            dtype=jnp.bool_
        )
        map_aug = lax.dynamic_update_slice(
            map_aug,
            map,
            start_indices=(LawnMowingFunctional.r_obs, LawnMowingFunctional.r_obs)
        )
        obs = lax.dynamic_slice(
            map_aug,
            start_indices=(y, x),
            slice_sizes=(2 * LawnMowingFunctional.r_obs, 2 * LawnMowingFunctional.r_obs)
        )
        return obs

    @staticmethod
    @jax.jit
    def crop_obs_pbc(map: jax.Array, x: int, y: int) -> jax.Array:
        map_aug = jnp.roll(
            map,
            shift=(LawnMowingFunctional.map_height // 2 - y, LawnMowingFunctional.map_width // 2 - x),
            axis=(0, 1)
        )
        obs = lax.dynamic_slice(
            map_aug,
            start_indices=(
                LawnMowingFunctional.map_height // 2 - LawnMowingFunctional.r_obs,
                LawnMowingFunctional.map_width // 2 - LawnMowingFunctional.r_obs
            ),
            slice_sizes=(2 * LawnMowingFunctional.r_obs, 2 * LawnMowingFunctional.r_obs)
        )
        return obs

    @staticmethod
    @jax.jit
    def crop_obs_rotate(map: jax.Array, x: int, y: int, theta: jax.Array, pad_ones: bool = True) -> jax.Array:
        map_aug = jnp.full(
            [LawnMowingFunctional.map_height + 2 * LawnMowingFunctional.diag_obs,
             LawnMowingFunctional.map_width + 2 * LawnMowingFunctional.diag_obs],
            fill_value=pad_ones,
            dtype=jnp.bool_
        )
        map_aug = lax.dynamic_update_slice(
            map_aug,
            map,
            start_indices=(LawnMowingFunctional.diag_obs, LawnMowingFunctional.diag_obs)
        )
        obs_aug = lax.dynamic_slice(
            map_aug,
            start_indices=(y, x),
            slice_sizes=(2 * LawnMowingFunctional.diag_obs, 2 * LawnMowingFunctional.diag_obs)
        )
        # Transform 2d bool array into 3d float array, meeting plx demands
        obs_aug = lax.broadcast(obs_aug, sizes=[1]).transpose(1, 2, 0).astype(jnp.float32)
        obs_aug = rotate_nearest(
            image=obs_aug,
            angle=theta[0] - jnp.pi,
            # mode='constant',
        )
        obs_aug = obs_aug.squeeze(axis=-1)
        obs = lax.dynamic_slice(
            obs_aug,
            start_indices=(LawnMowingFunctional.diag_obs - LawnMowingFunctional.r_obs,
                           LawnMowingFunctional.diag_obs - LawnMowingFunctional.r_obs),
            slice_sizes=(2 * LawnMowingFunctional.r_obs, 2 * LawnMowingFunctional.r_obs)
        )
        return obs

    @staticmethod
    @jax.jit
    def crop_obs_rotate_pbc(map: jax.Array, x: int, y: int, theta: jax.Array) -> jax.Array:
        map_aug = jnp.roll(
            map,
            shift=(LawnMowingFunctional.map_height // 2 - y, LawnMowingFunctional.map_width // 2 - x),
            axis=(0, 1)
        )
        obs_aug = lax.dynamic_slice(
            map_aug,
            start_indices=(
                LawnMowingFunctional.map_height // 2 - LawnMowingFunctional.diag_obs,
                LawnMowingFunctional.map_width // 2 - LawnMowingFunctional.diag_obs
            ),
            slice_sizes=(2 * LawnMowingFunctional.diag_obs, 2 * LawnMowingFunctional.diag_obs)
        )
        # Transform 2d bool array into 3d float array, meeting plx demands
        obs_aug = lax.broadcast(obs_aug, sizes=[1]).transpose(1, 2, 0).astype(jnp.float32)
        obs_aug = rotate_nearest(
            image=obs_aug,
            angle=theta[0] - jnp.pi,
            # mode='constant',
        )
        obs_aug = obs_aug.squeeze(axis=-1)
        obs = lax.dynamic_slice(
            obs_aug,
            start_indices=(LawnMowingFunctional.diag_obs - LawnMowingFunctional.r_obs,
                           LawnMowingFunctional.diag_obs - LawnMowingFunctional.r_obs),
            slice_sizes=(2 * LawnMowingFunctional.r_obs, 2 * LawnMowingFunctional.r_obs)
        )
        return obs

    def observation(self, state: EnvState) -> Dict[str, jax.Array]:
        """Cartpole observation."""
        x, y = state.position.round().astype(jnp.int32)
        cos_theta = jnp.cos(state.theta)
        sin_theta = jnp.sin(state.theta)
        pose = jnp.array([cos_theta, sin_theta]).squeeze(axis=1)
        if self.rotate_obs:
            if self.pbc:
                # Frontier
                obs_frontier = self.crop_obs_rotate_pbc(
                    map=state.map_frontier,
                    x=x,
                    y=y,
                    theta=state.theta,
                )
                # Obstacle
                obs_obstacle = self.crop_obs_rotate_pbc(
                    map=state.map_obstacle,
                    x=x,
                    y=y,
                    theta=state.theta,
                )
            else:
                # Frontier
                obs_frontier = self.crop_obs_rotate(
                    map=state.map_frontier,
                    x=x,
                    y=y,
                    theta=state.theta,
                    pad_ones=False,
                )
                # Obstacle
                obs_obstacle = self.crop_obs_rotate(
                    map=state.map_obstacle,
                    x=x,
                    y=y,
                    theta=state.theta,
                    pad_ones=True,
                )
        else:
            if self.pbc:
                # Frontier
                obs_frontier = self.crop_obs_pbc(
                    map=state.map_frontier,
                    x=x,
                    y=y,
                )
                # Obstacle
                obs_obstacle = self.crop_obs_pbc(
                    map=state.map_obstacle,
                    x=x,
                    y=y,
                )
            else:
                # Frontier
                obs_frontier = self.crop_obs(
                    map=state.map_frontier,
                    x=x,
                    y=y,
                    pad_ones=False,
                )
                # Obstacle
                obs_obstacle = self.crop_obs(
                    map=state.map_obstacle,
                    x=x,
                    y=y,
                    pad_ones=True,
                )
        obs = jnp.stack([obs_frontier, obs_obstacle], dtype=jnp.float32)
        obs_dict = {'observation': obs}
        if self.save_pixels:
            obs_dict['pixels'] = self.get_render(state)
        if not self.rotate_obs:
            obs_dict['pose'] = pose
        return obs_dict

    def terminal(self, state: EnvState) -> bool:
        """Checks if the state is terminal."""
        # terminated = jnp.logical_or(crashed, timestep >= self.max_timestep)
        terminated = state.timestep >= self.max_timestep
        return terminated

    def reward(
            self, state: EnvState, action: jax.Array, next_state: EnvState
    ) -> jax.Array:
        """Computes the reward for the state transition using the action."""
        reward_const = -0.1
        reward_collision = lax.select(next_state.crashed, -10, 0)

        map_area = self.map_width * self.map_height
        coverage_t = map_area - state.map_frontier.sum()
        coverage_tp1 = map_area - next_state.map_frontier.sum()
        reward_coverage = (coverage_tp1 - coverage_t) / (2 * self.r_self * self.v_max * self.tau) * 2
        # coverage_discount = self.r_self * self.v_max * self.tau / 2
        # reward_coverage = reward_coverage - coverage_discount
        # reward_coverage = lax.select(reward_coverage == 0, -7, 0)
        # v_linear, v_angular = self.get_velocity(action)
        # reward_steer = -jnp.power(v_angular * 10, 2) / 40

        tv_t = utils.total_variation(state.map_frontier.astype(dtype=jnp.int32))
        tv_tp1 = utils.total_variation(next_state.map_frontier.astype(dtype=jnp.int32))
        # reward_tv_global = -tv_t / jnp.sqrt(coverage_t)
        reward_tv_incremental = -(tv_tp1 - tv_t) / (2 * self.v_max * self.tau)

        reward = (
                reward_const
                + reward_collision
                + reward_coverage
                + reward_tv_incremental
            # + reward_steer
            # + reward_tv_global
        )
        return reward

    def render_image(
            self,
            state: EnvState,
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

        img = self.get_render(state)
        img = img.repeat(5, axis=0).repeat(5, axis=1)
        img = jax_to_numpy(img)

        surf = pygame.surfarray.make_surface(img)
        # surf = pygame.transform.flip(surf, False, True)

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

    @staticmethod
    @jax.jit
    def get_render(
            state: EnvState
    ) -> jax.Array:
        # # Mask for obs rectangle
        x, y = state.position.round().astype(jnp.int32)

        # obs_cols = lax.broadcast(
        #     jnp.arange(0, self.map_width),
        #     sizes=[self.map_height]
        # )
        # obs_rows = lax.broadcast(
        #     jnp.arange(0, self.map_height),
        #     sizes=[self.map_width]
        # ).swapaxes(0, 1)
        # mask_cols_range = jnp.logical_or(
        #     obs_cols == x - self.r_obs,
        #     obs_cols == x + self.r_obs
        # )
        # mask_rows_range = jnp.logical_or(
        #     obs_rows == y - self.r_obs,
        #     obs_rows == y + self.r_obs
        # )
        # mask_cols_condition = jnp.logical_and(
        #     obs_rows >= y - self.r_obs,
        #     obs_rows <= y + self.r_obs
        # )
        # mask_rows_condition = jnp.logical_and(
        #     obs_cols >= x - self.r_obs,
        #     obs_cols <= x + self.r_obs,
        # )
        # mask_cols = jnp.logical_and(
        #     mask_cols_range,
        #     mask_cols_condition
        # )
        # mask_rows = jnp.logical_and(
        #     mask_rows_range,
        #     mask_rows_condition,
        # )
        # mask_obs = jnp.logical_or(
        #     mask_cols,
        #     mask_rows
        # )
        # TV visualize
        mask_tv_cols = state.map_frontier.astype(jnp.uint8)[1:, :] - state.map_frontier.astype(jnp.uint8)[:-1, :] != 0
        mask_tv_cols = jnp.pad(mask_tv_cols, pad_width=[[0, 1], [0, 0]], mode='constant')
        mask_tv_rows = state.map_frontier.astype(jnp.uint8)[:, 1:] - state.map_frontier.astype(jnp.uint8)[:, :-1] != 0
        mask_tv_rows = jnp.pad(mask_tv_rows, pad_width=[[0, 0], [0, 1]], mode='constant')
        mask_tv = jnp.logical_or(mask_tv_cols, mask_tv_rows)
        # Draw covered area and agent
        img = jnp.ones([LawnMowingFunctional.map_height, LawnMowingFunctional.map_width, 3], dtype=jnp.uint8) * 255
        img = jnp.where(
            lax.broadcast(state.map_frontier, sizes=[3]).transpose(1, 2, 0) == 0,
            jnp.array([65, 227, 72], dtype=jnp.uint8),
            img
        )
        img = jnp.where(
            lax.broadcast(
                jnp.roll(
                    LawnMowingFunctional.vision_mask,
                    shift=(y - LawnMowingFunctional.map_height // 2, x - LawnMowingFunctional.map_width // 2),
                    axis=(0, 1)
                ),
                sizes=[3]
            ).transpose(1, 2, 0),
            jnp.array([255, 0, 0], dtype=jnp.uint8),
            img
        )
        # img = jnp.where(
        #     lax.broadcast(mask_obs, sizes=[3]).transpose(1, 2, 0),
        #     jnp.array([0, 0, 255], dtype=jnp.uint8),
        #     img
        # )
        img = jnp.where(
            lax.broadcast(mask_tv, sizes=[3]).transpose(1, 2, 0),
            jnp.array([255, 38, 255], dtype=jnp.uint8),
            img
        )
        # Scale up the img
        img = img.transpose(1, 0, 2)
        return img


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
