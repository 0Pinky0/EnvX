"""Implementation of a Jax-accelerated cartpole environment."""
from __future__ import annotations

from pathlib import Path
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

from envx.utils.total_variation import total_variation
from envx.utils.pix.jitted import rotate_nearest


class EnvState(NamedTuple):
    position: jax.Array
    theta: jax.Array
    # Channels: [Frontier(unseen), Obstacles, Farmland, Trajectory]
    map_frontier: jax.Array
    map_obstacle: jax.Array
    map_weed: jax.Array
    map_trajectory: jax.Array
    crashed: bool
    timestep: int
    init_map: jax.Array


RenderStateType = Tuple["pygame.Surface", "pygame.time.Clock"]


class PastureFunctional(
    FuncEnv[EnvState, jax.Array, jax.Array, float, bool, RenderStateType]
):
    tau: float = 0.5

    r_self: int = 4
    r_obs: int = 64
    r_vision: int = 24
    diag_obs: int = np.ceil(np.sqrt(2) * r_obs).astype(np.int32)

    max_timestep: int = 1000

    map_width = 600
    map_height = map_width

    screen_width = 600
    screen_height = 600

    v_max = 7.0
    w_max = 1.0
    # nvec = [4, 9]
    nvec = [7, 21]

    self_mask = (
                        (lax.broadcast(jnp.arange(0, 2 * r_obs), sizes=[2 * r_obs]) - r_obs) ** 2
                        + (lax.broadcast(jnp.arange(0, 2 * r_obs), sizes=[2 * r_obs]).swapaxes(0, 1)
                           - r_obs) ** 2
                ) <= r_self ** 2
    vision_mask = (
                          (lax.broadcast(jnp.arange(0, 2 * r_obs), sizes=[2 * r_obs]) - r_obs) ** 2
                          + (lax.broadcast(jnp.arange(0, 2 * r_obs), sizes=[2 * r_obs]).swapaxes(0, 1)
                             - r_obs) ** 2
                  ) <= r_vision ** 2

    num_obstacle_min = 3
    num_obstacle_max = 5

    obstacle_circle_radius_min = 8
    obstacle_circle_radius_max = 15

    pasture_ratio = 0.002
    decay_factor = 0.9

    farmland_map_num = 52
    # farmland_maps = jnp.load(f'{str(Path(__file__).parent.absolute())}/farmland_shapes/farmland_300.npy')
    farmland_maps = jnp.load(
        f'{str(Path(__file__).parent.parent.parent.absolute())}/data/farmland_shapes/farmland_{map_width}.npy')

    def __init__(
            self,
            save_pixels: bool = False,
            action_type: str = "continuous",
            rotate_obs: bool = False,
            prevent_stiff: bool = False,
            round_vision: bool = False,
            **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.save_pixels = save_pixels
        self.action_type = action_type
        self.rotate_obs = rotate_obs
        self.prevent_stiff = prevent_stiff
        self.round_vision = round_vision
        # Channels: [Frontier(unseen), Obstacles]
        # Future: [Farmland, Trajectory]
        # Define obs space
        obs_dict = {
            "observation": gym.spaces.Box(
                low=0.,
                high=1.,
                shape=(4, 2 * self.r_obs, 2 * self.r_obs),
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

    def initial(self, rng: PRNGKey):
        """Initial state generation."""
        x = jax.random.uniform(
            key=rng, minval=5 * self.r_self, maxval=self.map_height - 5 * self.r_self
        )
        _, rng = jax.random.split(rng)
        y = jax.random.uniform(
            key=rng, minval=5 * self.r_self, maxval=self.map_width - 5 * self.r_self
        )
        _, rng = jax.random.split(rng)
        position = jnp.stack([x, y])
        theta = jax.random.uniform(key=rng, minval=-jnp.pi, maxval=jnp.pi, shape=[1])

        x, y = position
        map_id = jax.random.randint(key=rng, shape=[1, ], minval=0, maxval=51)[0]
        _, rng = jax.random.split(rng)
        map_frontier = self.farmland_maps[map_id]
        # map_frontier = jnp.ones([self.map_height, self.map_width], dtype=jnp.bool_)
        new_vision_mask = (
                                  (lax.broadcast(jnp.arange(0, self.map_width),
                                                 sizes=[self.map_height]) - x) ** 2
                                  + (lax.broadcast(jnp.arange(0, self.map_height),
                                                   sizes=[self.map_width]).swapaxes(0, 1)
                                     - y) ** 2
                          ) <= self.r_self ** 2
        map_frontier = jnp.where(
            new_vision_mask,
            False,
            map_frontier,
        )

        num_obstacles = jax.random.randint(
            key=rng,
            shape=[1, ],
            minval=self.num_obstacle_min,
            maxval=self.num_obstacle_max
        )[0]
        _, rng = jax.random.split(rng)

        map_obstacle = jnp.zeros([self.map_height, self.map_width], dtype=jnp.bool_)

        def fill_obstacle(
                val: Tuple[jax.Array, jax.Array, jax.Array, jax.Array]
        ):
            map_frontier, map_obstacle, obstacle_mask, floor_mask = val
            map_frontier = jnp.where(
                floor_mask,
                False,
                map_frontier
            )
            map_obstacle = jnp.logical_or(map_obstacle, obstacle_mask)
            return map_frontier, map_obstacle

        def fill_obstacles(_, val: Tuple[jax.Array, jax.Array, jax.Array, PRNGKey]):
            map_frontier, map_obstacle, position, rng = val
            x, y = position
            o_x = jax.random.randint(
                key=rng,
                shape=[1, ],
                minval=0,
                maxval=self.map_height
            )[0]
            _, rng = jax.random.split(rng)
            o_y = jax.random.randint(
                key=rng,
                shape=[1, ],
                minval=0,
                maxval=self.map_width
            )[0]
            _, rng = jax.random.split(rng)
            o_r = jax.random.randint(
                key=rng,
                shape=[1, ],
                minval=self.obstacle_circle_radius_min,
                maxval=self.obstacle_circle_radius_max
            )[0]
            _, rng = jax.random.split(rng)
            obstacle_mask = (
                                    (lax.broadcast(jnp.arange(0, self.map_width),
                                                   sizes=[self.map_height]) - o_x) ** 2
                                    + (lax.broadcast(jnp.arange(0, self.map_height),
                                                     sizes=[self.map_width]).swapaxes(0, 1)
                                       - o_y) ** 2
                            ) <= o_r ** 2
            # Check if obstacle and agent are stacked
            agent_mask = (
                                 (lax.broadcast(jnp.arange(0, self.map_width),
                                                sizes=[self.map_height]) - x) ** 2
                                 + (lax.broadcast(jnp.arange(0, self.map_height),
                                                  sizes=[self.map_width]).swapaxes(0, 1)
                                    - y) ** 2
                         ) <= self.r_self ** 2
            floor_mask = (
                                 (lax.broadcast(jnp.arange(0, self.map_width),
                                                sizes=[self.map_height]) - o_x) ** 2
                                 + (lax.broadcast(jnp.arange(0, self.map_height),
                                                  sizes=[self.map_width]).swapaxes(0, 1)
                                    - o_y) ** 2
                         ) <= (o_r + self.r_self + 1) ** 2
            map_frontier, map_obstacle = lax.cond(
                jnp.logical_and(obstacle_mask, agent_mask).any(),
                lambda kwargs: (kwargs[0], kwargs[1]),
                fill_obstacle,
                (map_frontier, map_obstacle, obstacle_mask, floor_mask)
            )
            return map_frontier, map_obstacle, position, rng

        map_frontier, map_obstacle, _, rng = lax.fori_loop(
            0,
            num_obstacles,
            fill_obstacles,
            (map_frontier, map_obstacle, position, rng)
        )

        map_pasture = jax.random.uniform(shape=(self.map_height, self.map_width), key=rng) <= self.pasture_ratio
        map_pasture = jnp.where(map_frontier, map_pasture, False)
        _, rng = jax.random.split(rng)

        state = EnvState(
            position=position,
            theta=theta,
            map_frontier=map_frontier,
            map_obstacle=map_obstacle,
            map_weed=map_pasture,
            map_trajectory=jnp.zeros([self.map_height, self.map_width], dtype=jnp.bool_),
            crashed=False,
            timestep=0,
            init_map=map_frontier,
        )
        return state

    def get_velocity(self, action) -> Tuple[float, float]:
        match self.action_type:
            case "continuous":
                v_linear, v_angular = action
            case "discrete":
                action_linear = action // self.nvec[1] + self.prevent_stiff
                linear_size = self.nvec[0] - 1 + self.prevent_stiff
                v_linear = self.v_max * action_linear / linear_size
                action_angular = action % self.nvec[1]
                angular_size = (self.nvec[1] - 1) // 2
                v_angular = self.w_max * (action_angular - angular_size) / angular_size
            case "multi_discrete":
                linear_size = self.nvec[0] - 1
                v_linear = self.v_max * action[0] / linear_size
                angular_size = (self.nvec[1] - 1) // 2
                v_angular = self.w_max * (action[1] - angular_size) / angular_size
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
        # Examine whether outbounds
        x_oob = (
                (x < 0)
                | (x > self.map_width)
        )
        y_oob = (
                (y < 0)
                | (y > self.map_height)
        )
        crashed = (
                x_oob
                | y_oob
        )
        # Out of bounds
        x = lax.clamp(0., x, float(self.map_width))
        y = lax.clamp(0., y, float(self.map_height))
        new_position = jnp.array([x, y])
        new_theta = state.theta + v_angular * self.tau
        new_theta = (new_theta + jnp.pi) % (2 * jnp.pi) - jnp.pi

        # Update Maps
        # Bresenham
        x1, y1 = new_position.round().astype(jnp.int32)
        x2, y2 = state.position.round().astype(jnp.int32)

        dx = jnp.abs(x2 - x1)
        dy = jnp.abs(y2 - y1)
        slope = dy > dx

        x1, y1, x2, y2 = lax.select(
            slope,
            jnp.array([y1, x1, y2, x2]),
            jnp.array([x1, y1, x2, y2]),
        )
        x1, y1, x2, y2 = lax.select(
            x1 > x2,
            jnp.array([x2, y2, x1, y1]),
            jnp.array([x1, y1, x2, y2]),
        )

        dx = jnp.abs(x2 - x1)
        dy = jnp.abs(y2 - y1)
        error = dx // 2
        ystep = lax.select(y1 < y2, 1, -1)

        def bresenham_body(x, val: Tuple[int, float, jax.Array, jax.Array, jax.Array, jax.Array, bool]):
            y, error, map_frontier, map_pasture, map_trajectory, last_position, last_crashed = val
            next_position = lax.select(slope, jnp.array([y, x]), jnp.array([x, y]))
            x_aim, y_aim = next_position
            next_agent_mask = (
                                      (lax.broadcast(jnp.arange(0, self.map_width),
                                                     sizes=[self.map_height]) - x_aim) ** 2
                                      + (lax.broadcast(jnp.arange(0, self.map_height),
                                                       sizes=[self.map_width]).swapaxes(0, 1)
                                         - y_aim) ** 2
                              ) <= self.r_self ** 2
            next_crashed = jnp.logical_and(next_agent_mask, state.map_obstacle).any()
            next_crashed = jnp.logical_or(last_crashed, next_crashed)
            next_position = lax.select(next_crashed, last_position, next_position)
            map_pasture = jnp.where(
                next_agent_mask,
                False,
                map_pasture,
            )
            next_vision_mask = (
                                       (lax.broadcast(jnp.arange(0, self.map_width),
                                                      sizes=[self.map_height]) - x_aim) ** 2
                                       + (lax.broadcast(jnp.arange(0, self.map_height),
                                                        sizes=[self.map_width]).swapaxes(0, 1)
                                          - y_aim) ** 2
                               ) <= self.r_vision ** 2
            map_frontier = jnp.where(
                next_vision_mask,
                False,
                map_frontier,
            )
            map_trajectory = map_trajectory.at[y_aim, x_aim].set(True)
            error -= dy
            y += lax.select(error < 0, ystep, 0)
            error += lax.select(error < 0, dx, 0)
            return y, error, map_frontier, map_pasture, map_trajectory, next_position, next_crashed

        map_frontier = state.map_frontier
        map_pasture = state.map_weed
        map_trajectory = state.map_trajectory
        _, _, map_frontier, map_pasture, map_trajectory, crashed_position, crashed_when_running = lax.fori_loop(
            x1,
            x2 + 1,
            bresenham_body,
            (y1, error, map_frontier, map_pasture, map_trajectory, state.position.round().astype(jnp.int32), False)
        )
        crashed = jnp.logical_or(crashed, crashed_when_running)
        new_position = lax.select(crashed_when_running, crashed_position.astype(jnp.float32), new_position)

        # Construct new state
        state = EnvState(
            position=new_position,
            theta=new_theta,
            map_frontier=map_frontier,
            map_obstacle=state.map_obstacle,
            map_weed=map_pasture,
            map_trajectory=map_trajectory,
            crashed=crashed,
            timestep=state.timestep + 1,
            init_map=state.init_map,
        )

        return state

    @staticmethod
    @jax.jit
    def crop_obs(map: jax.Array, x: int, y: int, pad_ones: bool = True) -> jax.Array:
        map_aug = jnp.full(
            [PastureFunctional.map_height + 2 * PastureFunctional.r_obs,
             PastureFunctional.map_width + 2 * PastureFunctional.r_obs],
            fill_value=pad_ones,
            dtype=map.dtype
        )
        map_aug = lax.dynamic_update_slice(
            map_aug,
            map,
            start_indices=(PastureFunctional.r_obs, PastureFunctional.r_obs)
        )
        obs = lax.dynamic_slice(
            map_aug,
            start_indices=(y, x),
            slice_sizes=(2 * PastureFunctional.r_obs, 2 * PastureFunctional.r_obs)
        )
        return obs

    @staticmethod
    @jax.jit
    def crop_obs_rotate(map: jax.Array, x: int, y: int, theta: jax.Array, pad_ones: bool = True) -> jax.Array:
        map_aug = jnp.full(
            [PastureFunctional.map_height + 2 * PastureFunctional.diag_obs,
             PastureFunctional.map_width + 2 * PastureFunctional.diag_obs],
            fill_value=pad_ones,
            dtype=map.dtype
        )
        map_aug = lax.dynamic_update_slice(
            map_aug,
            map,
            start_indices=(PastureFunctional.diag_obs, PastureFunctional.diag_obs)
        )
        obs_aug = lax.dynamic_slice(
            map_aug,
            start_indices=(y, x),
            slice_sizes=(2 * PastureFunctional.diag_obs, 2 * PastureFunctional.diag_obs)
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
            start_indices=(PastureFunctional.diag_obs - PastureFunctional.r_obs,
                           PastureFunctional.diag_obs - PastureFunctional.r_obs),
            slice_sizes=(2 * PastureFunctional.r_obs, 2 * PastureFunctional.r_obs)
        )
        return obs

    @staticmethod
    @jax.jit
    def weed_apf(src: jax.Array, decay_factor: float) -> jax.Array:
        src = src.astype(jnp.float32)
        src = src[jnp.newaxis, jnp.newaxis, :, :]
        dst = src
        one_mask = src != 0
        horizon = jnp.ceil(1 / (1 - decay_factor)).astype(jnp.int32)

        def apf_loop(
                _,
                val: tuple[jax.Array, jax.Array, jax.Array, float]
        ) -> tuple[jax.Array, jax.Array, jax.Array, float]:
            src, dst, one_mask, decay_factor = val
            kernel = jnp.array([
                [0., decay_factor, 0.],
                [decay_factor, 1., decay_factor],
                [0., decay_factor, 0.],
            ])[jnp.newaxis, jnp.newaxis, :, :]
            src = lax.conv(
                src,
                kernel,
                window_strides=(1, 1),
                padding='SAME'
            )
            src = lax.clamp(0., src, decay_factor)
            dst = jnp.where(
                jnp.logical_not(one_mask),
                src,
                dst,
            )
            one_mask = jnp.logical_or(src != 0, one_mask)
            src = one_mask.astype(jnp.float32)
            decay_factor *= decay_factor
            return src, dst, one_mask, decay_factor

        src, dst, one_mask, decay_factor = lax.fori_loop(0, horizon, apf_loop, (src, dst, one_mask, decay_factor))
        dst = dst[0, 0]
        return dst

    def observation(self, state: EnvState) -> Dict[str, jax.Array]:
        """Cartpole observation."""
        x, y = state.position.round().astype(jnp.int32)
        cos_theta = jnp.cos(state.theta)
        sin_theta = jnp.sin(state.theta)
        pose = jnp.array([cos_theta, sin_theta]).squeeze(axis=1)
        observed_weed = jnp.where(
            jnp.logical_not(state.map_frontier),
            state.map_weed,
            False,
        )
        # observed_weed = self.get_map_pasture_larger(observed_weed)
        observed_weed = self.weed_apf(src=observed_weed, decay_factor=self.decay_factor)
        if self.rotate_obs:
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
            # Pasture
            obs_weed = self.crop_obs_rotate(
                map=observed_weed,
                x=x,
                y=y,
                theta=state.theta,
                pad_ones=False,
            )
            obs_traj = self.crop_obs_rotate(
                map=state.map_trajectory,
                x=x,
                y=y,
                theta=state.theta,
                pad_ones=False,
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
            # Pasture
            obs_weed = self.crop_obs(
                map=observed_weed,
                x=x,
                y=y,
                pad_ones=False,
            )
            obs_traj = self.crop_obs(
                map=state.map_trajectory,
                x=x,
                y=y,
                pad_ones=False,
            )

        obs = jnp.stack(
            [
                obs_frontier,
                obs_obstacle,
                obs_weed,
                obs_traj
            ],
            dtype=jnp.float32
        )
        obs_dict = {'observation': obs}
        if self.save_pixels:
            obs_dict['pixels'] = self.get_render(state)
        if not self.rotate_obs:
            obs_dict['pose'] = pose
        return obs_dict

    def terminal(self, state: EnvState) -> bool:
        """Checks if the state is terminal."""
        judge_out_of_time = state.timestep >= self.max_timestep

        area_total = state.init_map.sum()
        area_covered = area_total - state.map_frontier.sum()
        coverage_ratio = area_covered / area_total
        judge_covered_enough = coverage_ratio >= 0.99

        pasture_covered_enough = state.map_weed.sum() == 0
        judge_covered_enough = jnp.logical_and(judge_covered_enough, pasture_covered_enough)

        terminated = jnp.logical_or(judge_covered_enough, judge_out_of_time)
        return terminated  # noqa: Jax traced type can be handled correctly

    def reward(
            self, state: EnvState, action: jax.Array, next_state: EnvState
    ) -> jax.Array:
        """Computes the reward for the state transition using the action."""
        reward_const = -0.1
        reward_collision = lax.select(next_state.crashed, -10, 0)

        # v_linear, v_angular = self.get_velocity(action)
        # v_linear, v_angular = v_linear / self.v_max, v_angular / self.w_max
        # reward_stiff = (lax.select(jnp.logical_and(v_linear == 0, v_angular == 0), -2.5, 0.)
        #                 + lax.select(v_linear == 0, -0.4, 0.))
        # reward_dynamic = -((jnp.abs(v_angular) / self.w_max) ** 2) / 2

        map_area = self.map_width * self.map_height
        coverage_t = map_area - state.map_frontier.sum()
        coverage_tp1 = map_area - next_state.map_frontier.sum()
        reward_coverage = (coverage_tp1 - coverage_t) / (2 * self.r_vision * self.v_max * self.tau) * 0.25
        num_pasture = state.map_weed.sum() - next_state.map_weed.sum()
        reward_pasture = num_pasture * 0.5

        tv_t = total_variation(state.map_frontier.astype(dtype=jnp.int32))
        tv_tp1 = total_variation(next_state.map_frontier.astype(dtype=jnp.int32))
        # reward_tv_global = -tv_t / jnp.sqrt(coverage_t)
        reward_tv_incremental = -(tv_tp1 - tv_t) / (self.v_max * self.tau) * 0.25

        reward = (
                reward_const
                + reward_collision
                # + reward_coverage
                # + reward_tv_incremental
                + reward_pasture
            # + reward_stiff
            # + reward_dynamic
            # + reward_steer
            # + reward_tv_global
        )
        # reward = reward / 10
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
        # img = img.repeat(5, axis=0).repeat(5, axis=1)
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
    def get_map_pasture_larger(map_pasture: jax.Array):
        map_pasture_larger = map_pasture
        map_pasture_larger = jnp.logical_or(
            map_pasture_larger,
            jnp.insert(
                map_pasture[1:, :],
                -1,
                False,
                axis=0
            )
        )
        map_pasture_larger = jnp.logical_or(
            map_pasture_larger,
            jnp.insert(
                map_pasture[:-1, :],
                0,
                False,
                axis=0
            )
        )
        map_pasture_larger = jnp.logical_or(
            map_pasture_larger,
            jnp.insert(
                map_pasture[:, 1:],
                -1,
                False,
                axis=1
            )
        )
        map_pasture_larger = jnp.logical_or(
            map_pasture_larger,
            jnp.insert(
                map_pasture[:, :-1],
                0,
                False,
                axis=1
            )
        )
        return map_pasture_larger

    @staticmethod
    @jax.jit
    def get_render(
            state: EnvState
    ) -> jax.Array:
        x, y = state.position.round().astype(jnp.int32)
        # TV visualize
        mask_tv_cols = state.map_frontier.astype(jnp.uint8)[1:, :] - state.map_frontier.astype(jnp.uint8)[:-1, :] != 0
        mask_tv_cols = jnp.pad(mask_tv_cols, pad_width=[[0, 1], [0, 0]], mode='constant')
        mask_tv_rows = state.map_frontier.astype(jnp.uint8)[:, 1:] - state.map_frontier.astype(jnp.uint8)[:, :-1] != 0
        mask_tv_rows = jnp.pad(mask_tv_rows, pad_width=[[0, 0], [0, 1]], mode='constant')
        mask_tv = jnp.logical_or(mask_tv_cols, mask_tv_rows)
        # Draw covered area and agent
        mask_covered = jnp.logical_and(state.init_map, state.map_frontier)
        mask_uncovered = jnp.logical_not(mask_covered)
        img = jnp.ones([PastureFunctional.map_height, PastureFunctional.map_width, 3], dtype=jnp.uint8) * 255
        ## Old render: all green
        # img = jnp.where(
        #     lax.broadcast(state.map_frontier, sizes=[3]).transpose(1, 2, 0) == 0,
        #     jnp.array([65, 227, 72], dtype=jnp.uint8),
        #     img
        # )
        new_vision_mask = (
                                  (lax.broadcast(jnp.arange(0, PastureFunctional.map_width),
                                                 sizes=[PastureFunctional.map_height]) - x) ** 2
                                  + (lax.broadcast(jnp.arange(0, PastureFunctional.map_height),
                                                   sizes=[PastureFunctional.map_width]).swapaxes(0, 1)
                                     - y) ** 2
                          ) <= PastureFunctional.r_vision ** 2
        img = jnp.where(
            lax.broadcast(new_vision_mask, sizes=[3]).transpose(1, 2, 0),
            jnp.array([192, 192, 192], dtype=jnp.uint8),
            img
        )
        ## New render: yellow farmland
        img = jnp.where(
            lax.broadcast(mask_uncovered, sizes=[3]).transpose(1, 2, 0) == 0,
            jnp.array([255, 215, 0], dtype=jnp.uint8),
            img
        )
        # Pasture
        map_pasture_larger = PastureFunctional.get_map_pasture_larger(state.map_weed)
        img = jnp.where(
            lax.broadcast(
                jnp.logical_and(mask_covered, map_pasture_larger),
                sizes=[3]
            ).transpose(1, 2, 0),
            jnp.array([255, 0, 0], dtype=jnp.uint8),
            img
        )
        img = jnp.where(
            lax.broadcast(
                jnp.logical_and(mask_uncovered, map_pasture_larger),
                sizes=[3]
            ).transpose(1, 2, 0),
            jnp.array([64, 255, 64], dtype=jnp.uint8),
            img
        )
        # Obstacles
        img = jnp.where(
            lax.broadcast(state.map_obstacle, sizes=[3]).transpose(1, 2, 0),
            jnp.array([128, 128, 128], dtype=jnp.uint8),
            img
        )
        # Trajectory
        img = jnp.where(
            lax.broadcast(state.map_trajectory, sizes=[3]).transpose(1, 2, 0),
            jnp.array([0, 128, 255], dtype=jnp.uint8),
            img
        )
        # if LawnMowingFunctional.pbc:
        #     new_vision_mask = jnp.roll(
        #         LawnMowingFunctional.vision_mask,
        #         shift=(y - LawnMowingFunctional.map_height // 2, x - LawnMowingFunctional.map_width // 2),
        #         axis=(0, 1)
        #     )
        # else:
        new_self_mask = (
                                (lax.broadcast(jnp.arange(0, PastureFunctional.map_width),
                                               sizes=[PastureFunctional.map_height]) - x) ** 2
                                + (lax.broadcast(jnp.arange(0, PastureFunctional.map_height),
                                                 sizes=[PastureFunctional.map_width]).swapaxes(0, 1)
                                   - y) ** 2
                        ) <= PastureFunctional.r_self ** 2
        # Agent head
        img = jnp.where(
            lax.broadcast(
                new_self_mask,
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


class PastureJaxEnv(FunctionalJaxEnv, EzPickle):
    """Jax-based implementation of the CartPole environment."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 50}

    def __init__(self, render_mode: str | None = None, **kwargs: Any):
        """Constructor for the CartPole where the kwargs are applied to the functional environment."""
        EzPickle.__init__(self, render_mode=render_mode, **kwargs)

        env = PastureFunctional(**kwargs)
        env.transform(jax.jit)

        FunctionalJaxEnv.__init__(
            self,
            env,
            metadata=self.metadata,
            render_mode=render_mode,
        )


class PastureJaxVectorEnv(FunctionalJaxVectorEnv, EzPickle):
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

        env = PastureFunctional(**kwargs)
        env.transform(jax.jit)

        FunctionalJaxVectorEnv.__init__(
            self,
            func_env=env,
            num_envs=num_envs,
            metadata=self.metadata,
            render_mode=render_mode,
            max_episode_steps=max_episode_steps,
        )
