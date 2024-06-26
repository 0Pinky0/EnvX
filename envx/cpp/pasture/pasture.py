"""Implementation of a Jax-accelerated cartpole environment."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Tuple, NamedTuple, Dict, Sequence

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
from envx.utils.apf_jax import apf


class EnvState(NamedTuple):
    position: jax.Array
    theta: jax.Array
    # Channels: [Frontier(unseen), Obstacles, Farmland, Trajectory]
    map_frontier: jax.Array
    map_obstacle: jax.Array
    map_weed: jax.Array
    observed_weed: jax.Array
    map_trajectory: jax.Array
    diff_trajectory: jax.Array
    crashed: bool
    crash_count: int
    timestep: int
    init_map: jax.Array
    weed_count: int
    traj_crossed: bool


RenderStateType = Tuple["pygame.Surface", "pygame.time.Clock"]


def to_left(p: tuple[int, int], q: tuple[int, int], s_x: jax.Array, s_y: jax.Array) -> jax.Array:
    p_x, p_y = p
    q_x, q_y = q
    return (0 < p_x * q_y - p_y * q_x
            + q_x * s_y - q_y * s_x
            + s_x * p_y - s_y * p_x)


def in_triangle(z_1: tuple[int, int], z_2: tuple[int, int], z_3: tuple[int, int], r: int) -> jax.Array:
    s_x = lax.broadcast(jnp.arange(0, 2 * r + 1), sizes=[2 * r + 1])
    s_y = lax.broadcast(jnp.arange(0, 2 * r + 1), sizes=[2 * r + 1]).swapaxes(0, 1)
    return jnp.logical_and(
        to_left(z_1, z_2, s_x, s_y) == to_left(z_2, z_3, s_x, s_y),
        to_left(z_2, z_3, s_x, s_y) == to_left(z_3, z_1, s_x, s_y),
    )


class PastureFunctional(
    FuncEnv[EnvState, jax.Array, jax.Array, float, bool, RenderStateType]
):
    tau: float = 0.5

    r_self: int = 4
    r_obs: int = 64
    r_vision: int = 24

    max_timestep: int = 2_000

    map_width = 300
    map_height = map_width

    screen_width = 600
    screen_height = 600

    v_max = 7.0
    w_max = 1.0
    # nvec = [4, 9]
    nvec = [7, 21]

    decay_factor = 0.9
    decay_lowerbound = 1e-2

    triangle_obs: bool = True
    w_vision = 24

    # Supporting variables, need not change
    sgcnn_scale = 4
    sgcnn_size = 2 * r_obs // (2 ** (sgcnn_scale - 1))
    diag_obs: int = np.ceil(np.sqrt(2) * r_obs).astype(np.int32)
    vision_mask = (
                          (lax.broadcast(jnp.arange(0, 2 * r_vision + 1), sizes=[2 * r_vision + 1]) - r_vision) ** 2
                          + (lax.broadcast(jnp.arange(0, 2 * r_vision + 1), sizes=[2 * r_vision + 1]).swapaxes(0, 1)
                             - r_vision) ** 2
                  ) <= r_vision ** 2
    # frontier_maps = jnp.load(
    #     f'{str(Path(__file__).parent.parent.parent.absolute())}/data/farmland_v2/farmland_v2_1.npy',
    #     mmap_mode='r')
    frontier_maps = jnp.load(
        f'{str(Path(__file__).parent.parent.parent.absolute())}/data/farmland_shapes/farmland_{map_width}.npy',
        mmap_mode='r')
    if triangle_obs:
        triangle_mask = in_triangle(
            z_1=(r_vision, r_vision),
            z_2=(r_vision - w_vision, 0),
            z_3=(r_vision + w_vision, 0),
            r=r_vision,
        )
        vision_mask = jnp.where(
            triangle_mask,
            vision_mask,
            False
        )

    def __init__(
            self,
            save_pixels: bool = False,
            return_map: bool = False,
            action_type: str = "continuous",
            rotate_obs: bool = False,
            prevent_stiff: bool = False,
            sgcnn: bool = False,
            global_obs: bool = False,
            use_traj: bool = False,
            diff_traj: bool = False,
            use_apf: bool = False,
            gaussian_weed: bool = False,
            weed_count: int = None,
            weed_ratio: float = 0.002,
            map_id: int = None,
            num_obstacle_min: int = 3,
            num_obstacle_max: int = 5,
            obstacle_circle_radius_min: int = 8,
            obstacle_circle_radius_max: int = 15,
            return_info: bool = False,
            **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.save_pixels = save_pixels
        self.action_type = action_type
        self.rotate_obs = rotate_obs
        self.prevent_stiff = prevent_stiff
        self.use_traj = use_traj
        self.diff_traj = diff_traj
        self.sgcnn = sgcnn
        self.global_obs = global_obs
        self.use_apf = use_apf
        self.return_map = return_map
        self.gaussian_weed = gaussian_weed
        self.map_id = map_id
        self.num_obstacle_min = num_obstacle_min
        self.num_obstacle_max = num_obstacle_max
        self.obstacle_circle_radius_min = obstacle_circle_radius_min
        self.obstacle_circle_radius_max = obstacle_circle_radius_max
        self.return_info = return_info
        if weed_count is not None:
            self.weed_ratio = weed_count / (self.map_height * self.map_width)
        else:
            self.weed_ratio = weed_ratio
        # Channels: [Frontier(unseen), Obstacles]
        # Future: [Farmland, Trajectory]
        # Define obs space
        num_channels = 4 if self.use_traj else 3
        obs_dict = {}
        if return_map:
            obs_dict["map"] = gym.spaces.Box(
                low=0.,
                high=1.,
                shape=(num_channels, self.map_height, self.map_width),
                dtype=np.float32
            )
        else:
            obs_dict["observation"] = gym.spaces.Box(
                low=0.,
                high=1.,
                shape=((self.sgcnn_scale + self.global_obs) * num_channels if self.sgcnn else num_channels,
                       self.sgcnn_size if self.sgcnn else 2 * self.r_obs,
                       self.sgcnn_size if self.sgcnn else 2 * self.r_obs),
                dtype=np.float32
            )
        if save_pixels:
            obs_dict["pixels"] = gym.spaces.Box(
                low=0,
                high=255,
                shape=(self.map_height, self.map_width, 3),
                dtype=np.uint8
            )
        if not rotate_obs:
            obs_dict["pose"] = gym.spaces.Box(low=-1., high=1., shape=(2,), dtype=np.float32)
        if return_map:
            obs_dict["position"] = gym.spaces.Box(low=0., high=self.map_width, shape=(2,), dtype=np.float32)
            obs_dict["theta"] = gym.spaces.Box(low=-np.pi, high=np.pi, shape=(1,), dtype=np.float32)
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
        # position = jnp.stack([4., 4.])
        # theta = jnp.array([jnp.pi / 2])

        x, y = position.round().astype(jnp.int32)
        if self.map_id is not None:
            map_id = self.map_id
        else:
            map_id = jax.random.randint(key=rng, shape=[1, ], minval=0, maxval=self.frontier_maps.shape[0] - 1)[0]
            _, rng = jax.random.split(rng)
        map_frontier = self.frontier_maps[map_id]
        # map_frontier = jnp.load(
        #     f'{str(Path(__file__).parent.parent.parent.absolute())}/data/farmland_v2/1/farmland_{map_id}.npy')
        # map_frontier = self.farmland_maps[map_id]
        # map_frontier = jnp.ones([self.map_height, self.map_width], dtype=jnp.bool_)
        # new_vision_mask = (
        #                           (lax.broadcast(jnp.arange(0, self.map_width),
        #                                          sizes=[self.map_height]) - x) ** 2
        #                           + (lax.broadcast(jnp.arange(0, self.map_height),
        #                                            sizes=[self.map_width]).swapaxes(0, 1)
        #                              - y) ** 2
        #                   ) <= self.r_self ** 2
        new_vision_mask = jnp.zeros([
            self.map_height + 2 * self.r_vision,
            self.map_width + 2 * self.r_vision],
            dtype=jnp.bool_)
        if self.triangle_obs:
            rotated_vision_mask = rotate_nearest(
                image=lax.broadcast(self.vision_mask, sizes=[1, ]).transpose(1, 2, 0),
                angle=-(jnp.pi / 2 + theta[0]),
                # mode='constant',
            ).transpose(2, 0, 1)[0]
        else:
            rotated_vision_mask = self.vision_mask
        new_vision_mask = lax.dynamic_update_slice(new_vision_mask,
                                                   rotated_vision_mask,
                                                   start_indices=(y, x))
        new_vision_mask = lax.dynamic_slice(new_vision_mask,
                                            start_indices=(self.r_vision, self.r_vision),
                                            slice_sizes=(self.map_height, self.map_width))

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

        if self.gaussian_weed:
            map_weed = jax.random.normal(shape=(self.map_height, self.map_width), key=rng)
            map_weed = jnp.where(
                map_frontier,
                map_weed,
                100.0,
            )
            discrete_num = int(self.map_height * self.map_width * self.weed_ratio) - 1
            desired_ratio = map_weed.flatten().sort()[discrete_num]
            map_weed = map_weed <= desired_ratio
        else:
            map_weed = jax.random.uniform(shape=(self.map_height, self.map_width), key=rng)
            map_weed = jnp.where(
                map_frontier,
                map_weed,
                100.0,
            )
            discrete_num = int(self.map_height * self.map_width * self.weed_ratio) - 1
            desired_ratio = map_weed.flatten().sort()[discrete_num]
            map_weed = map_weed <= desired_ratio
        _, rng = jax.random.split(rng)

        observed_weed = jnp.where(
            jnp.logical_not(map_frontier),
            map_weed,
            False,
        )
        if self.use_apf:
            observed_weed = apf(observed_weed)
            observed_weed = lax.select(observed_weed.sum() == 0., observed_weed, self.decay_factor ** observed_weed)
            observed_weed = jnp.where(observed_weed < self.decay_lowerbound, 0., observed_weed)

        state = EnvState(
            position=position,
            theta=theta,
            map_frontier=map_frontier,
            map_obstacle=map_obstacle,
            map_weed=map_weed,
            observed_weed=observed_weed,
            map_trajectory=jnp.zeros([self.map_height, self.map_width], dtype=jnp.uint8),
            diff_trajectory=jnp.zeros([self.map_height, self.map_width], dtype=jnp.uint8),
            crashed=False,
            crash_count=0,
            timestep=0,
            init_map=map_frontier,
            weed_count=map_weed.sum(),
            traj_crossed=False,
        )
        return state

    def get_velocity(self, action: jax.Array) -> Tuple[jax.Array, jax.Array]:
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

        if self.triangle_obs:
            rotated_vision_mask = rotate_nearest(
                image=lax.broadcast(self.vision_mask, sizes=[1, ]).transpose(1, 2, 0),
                angle=-(jnp.pi / 2 + state.theta[0]),
                # mode='constant',
            ).transpose(2, 0, 1)[0]
        else:
            rotated_vision_mask = self.vision_mask

        def bresenham_body(x, val: Tuple[int, float, jax.Array, jax.Array, jax.Array, jax.Array, bool, bool, bool]):
            y, error, map_frontier, map_pasture, map_trajectory, last_position, last_crashed, traj_crossed, is_first = val
            next_position = lax.select(slope, jnp.array([y, x]), jnp.array([x, y]))
            x_aim, y_aim = next_position
            # Agent's coverage mask in next step
            next_agent_mask = (
                                      (lax.broadcast(jnp.arange(0, self.map_width),
                                                     sizes=[self.map_height]) - x_aim) ** 2
                                      + (lax.broadcast(jnp.arange(0, self.map_height),
                                                       sizes=[self.map_width]).swapaxes(0, 1)
                                         - y_aim) ** 2
                              ) <= self.r_self ** 2
            # next_agent_mask = jnp.zeros([
            #     self.map_height + 2 * self.r_self,
            #     self.map_width + 2 * self.r_self],
            #     dtype=jnp.bool_)
            # next_agent_mask = lax.dynamic_update_slice(next_agent_mask,
            #                                            self.self_mask,
            #                                            start_indices=(y_aim, x_aim))
            # next_agent_mask = lax.dynamic_slice(next_agent_mask,
            #                                     start_indices=(self.r_self, self.r_self),
            #                                     slice_sizes=(self.map_height, self.map_width))
            next_crashed = jnp.logical_and(next_agent_mask, state.map_obstacle).any()
            next_crashed = jnp.logical_or(last_crashed, next_crashed)
            next_position = lax.select(next_crashed, last_position, next_position)
            map_pasture = jnp.where(
                next_agent_mask,
                False,
                map_pasture,
            )
            # next_vision_mask = (
            #                            (lax.broadcast(jnp.arange(0, self.map_width),
            #                                           sizes=[self.map_height]) - x_aim) ** 2
            #                            + (lax.broadcast(jnp.arange(0, self.map_height),
            #                                             sizes=[self.map_width]).swapaxes(0, 1)
            #                               - y_aim) ** 2
            #                    ) <= self.r_vision ** 2
            next_vision_mask = jnp.zeros([
                self.map_height + 2 * self.r_vision,
                self.map_width + 2 * self.r_vision],
                dtype=jnp.bool_)
            next_vision_mask = lax.dynamic_update_slice(next_vision_mask,
                                                        rotated_vision_mask,
                                                        start_indices=(y_aim, x_aim))
            next_vision_mask = lax.dynamic_slice(next_vision_mask,
                                                 start_indices=(self.r_vision, self.r_vision),
                                                 slice_sizes=(self.map_height, self.map_width))
            map_frontier = jnp.where(
                next_vision_mask,
                False,
                map_frontier,
            )
            map_trajectory = map_trajectory.at[y_aim, x_aim].set(map_trajectory[y_aim, x_aim] + 1)
            has_cross = lax.select(state.map_trajectory[y_aim, x_aim] > 0, True, False)
            is_first = jnp.logical_and(is_first, has_cross)
            traj_crossed = jnp.logical_and(has_cross, jnp.logical_not(is_first))
            error -= dy
            y += lax.select(error < 0, ystep, 0)
            error += lax.select(error < 0, dx, 0)
            return y, error, map_frontier, map_pasture, map_trajectory, next_position, next_crashed, traj_crossed, is_first

        map_frontier = state.map_frontier
        map_weed = state.map_weed
        map_trajectory = state.map_trajectory
        _, _, map_frontier, map_weed, map_trajectory, crashed_position, crashed_when_running, traj_crossed, _ = lax.fori_loop(
            x1,
            x2 + 1,
            bresenham_body,
            (y1, error, map_frontier, map_weed, map_trajectory, state.position.round().astype(jnp.int32), False, False,
             True)
        )
        crashed = jnp.logical_or(crashed, crashed_when_running)
        new_position = lax.select(crashed_when_running, crashed_position.astype(jnp.float32), new_position)

        observed_weed = jnp.where(
            jnp.logical_not(map_frontier),
            map_weed,
            False,
        )
        diff_trajectory = map_trajectory != 0
        if self.use_apf:
            observed_weed = apf(observed_weed)
            observed_weed = lax.select(observed_weed.sum() == 0., observed_weed, self.decay_factor ** observed_weed)
            observed_weed = jnp.where(observed_weed < self.decay_lowerbound, 0., observed_weed)
        if self.diff_traj:
            diff_trajectory = apf(diff_trajectory)
            diff_trajectory = lax.select(diff_trajectory.sum() == 0., diff_trajectory,
                                        self.decay_factor ** diff_trajectory)
            diff_trajectory = jnp.where(diff_trajectory < self.decay_factor ** 2.83, 0., diff_trajectory)

        crash_count = lax.select(crashed, state.crash_count + 1, 0)

        # Construct new state
        state = EnvState(
            position=new_position,
            theta=new_theta,
            map_frontier=map_frontier,
            map_obstacle=state.map_obstacle,
            map_weed=map_weed,
            observed_weed=observed_weed,
            map_trajectory=map_trajectory,
            diff_trajectory=diff_trajectory,
            crashed=crashed,
            crash_count=crash_count,
            timestep=state.timestep + 1,
            init_map=state.init_map,
            weed_count=state.weed_count,
            traj_crossed=traj_crossed,
        )
        return state

    def observation(self, state: EnvState) -> Dict[str, jax.Array]:
        """Cartpole observation."""
        x, y = state.position.round().astype(jnp.int32)
        obs_list = [state.map_frontier,
                    state.map_obstacle,
                    state.observed_weed]
        init_val = [0., 1., 0.]
        if self.use_traj:
            init_val.append(0.)
            obs_list.append(state.diff_trajectory)
        obs_full = jnp.stack(
            obs_list,
            dtype=jnp.float32
        )
        obs_dict = {}
        if self.save_pixels:
            obs_dict['pixels'] = self.get_render(state)
        if not self.rotate_obs:
            cos_theta = jnp.cos(state.theta)
            sin_theta = jnp.sin(state.theta)
            pose = jnp.array([cos_theta, sin_theta]).squeeze(axis=1)
            obs_dict['pose'] = pose
        if self.return_map:
            # obs_list.append(state.map_trajectory)
            # obs = jnp.stack(
            #     obs_list,
            #     dtype=jnp.float32
            # )
            obs_dict['map'] = obs_full
            obs_dict['position'] = state.position
            obs_dict['theta'] = state.theta
        else:
            num_channels = 4 if self.use_traj else 3
            if self.rotate_obs:
                obs_aug = jnp.full(
                    [num_channels,
                     self.map_height + 2 * self.diag_obs,
                     self.map_width + 2 * self.diag_obs],
                    fill_value=jnp.broadcast_to(jnp.array(init_val), shape=(1, 1, num_channels)).transpose(2, 0, 1),
                    dtype=jnp.float32
                )
                obs_aug = lax.dynamic_update_slice(
                    obs_aug,
                    obs_full,
                    start_indices=(0, self.diag_obs, self.diag_obs)
                )
                obs_aug = lax.dynamic_slice(
                    obs_aug,
                    start_indices=(0, y, x),
                    slice_sizes=(num_channels, 2 * self.diag_obs, 2 * self.diag_obs)
                )
                # Transform 2d bool array into 3d float array, meeting pix demands
                obs_aug = rotate_nearest(
                    image=obs_aug.transpose(1, 2, 0),
                    angle=state.theta[0] - jnp.pi,
                    # mode='constant',
                ).transpose(2, 0, 1)
                obs = lax.dynamic_slice(
                    obs_aug,
                    start_indices=(0,
                                   self.diag_obs - self.r_obs,
                                   self.diag_obs - self.r_obs),
                    slice_sizes=(num_channels, 2 * self.r_obs, 2 * self.r_obs)
                )
            else:
                obs_aug = jnp.full(
                    [num_channels,
                     self.map_height + 2 * self.r_obs,
                     self.map_width + 2 * self.r_obs],
                    fill_value=jnp.broadcast_to(jnp.array(init_val), shape=(1, 1, num_channels)).transpose(2, 0, 1),
                    dtype=jnp.float32
                )
                obs_aug = lax.dynamic_update_slice(
                    obs_aug,
                    obs_full,
                    start_indices=(0, self.r_obs, self.r_obs)
                )
                obs = lax.dynamic_slice(
                    obs_aug,
                    start_indices=(0, y, x),
                    slice_sizes=(num_channels, 2 * self.r_obs, 2 * self.r_obs)
                )
            if self.sgcnn:
                obs_ = obs
                obs_list = []
                for _ in range(self.sgcnn_scale):
                    obs_s = lax.dynamic_slice(
                        obs_,
                        start_indices=(
                            0, self.r_obs - self.sgcnn_size // 2, self.r_obs - self.sgcnn_size // 2),
                        slice_sizes=(num_channels, self.sgcnn_size, self.sgcnn_size)
                    )
                    obs_ = lax.reduce_window(obs_, -jnp.inf, lax.max, (1, 2, 2), (1, 2, 2), padding='VALID')
                    obs_list.append(obs_s)
                if self.global_obs:
                    kernel_size = int(np.ceil(2 * np.sqrt(2) * self.map_width / self.sgcnn_size))
                    obs_g = lax.reduce_window(
                        obs_full,
                        -jnp.inf,
                        lax.max,
                        (1, kernel_size, kernel_size),
                        (1, kernel_size, kernel_size),
                        padding='VALID'
                    )
                    g_size = obs_g.shape[-1]
                    obs_aug = jnp.full(
                        [num_channels,
                         self.sgcnn_size,
                         self.sgcnn_size],
                        fill_value=jnp.broadcast_to(jnp.array(init_val), shape=(1, 1, num_channels)).transpose(2, 0, 1),
                        dtype=jnp.float32
                    )
                    x, y = jnp.round(state.position / self.map_width * g_size).astype(jnp.int32)
                    obs_aug = lax.dynamic_update_slice(
                        obs_aug,
                        obs_g,
                        start_indices=(0, self.sgcnn_size // 2 - y, self.sgcnn_size // 2 - x)
                    )
                    # Transform 2d bool array into 3d float array, meeting pix demands
                    obs_g = rotate_nearest(
                        image=obs_aug.transpose(1, 2, 0),
                        angle=state.theta[0] - jnp.pi,
                        # mode='constant',
                    ).transpose(2, 0, 1)
                    obs_list.append(obs_g)
                obs = lax.concatenate(obs_list, dimension=0)
            obs_dict['observation'] = obs
        return obs_dict

    def terminal(self, state: EnvState) -> bool:
        """Checks if the state is terminal."""
        judge_out_of_time = state.timestep >= self.max_timestep

        # area_total = state.init_map.sum()
        # area_covered = area_total - state.map_frontier.sum()
        # coverage_ratio = area_covered / area_total
        # judge_covered_enough = coverage_ratio >= 0.99

        pasture_covered_enough = state.map_weed.sum() == 0
        judge_covered_enough = pasture_covered_enough
        # judge_covered_enough = jnp.logical_or(judge_covered_enough, pasture_covered_enough)

        crash_too_much = state.crash_count > 5

        terminated = jnp.logical_or(judge_covered_enough, judge_out_of_time)
        terminated = jnp.logical_or(terminated, crash_too_much)
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
        reward_coverage = (coverage_tp1 - coverage_t) / (2 * self.r_vision * self.v_max * self.tau) * 0.0075

        tv_t = total_variation(state.map_frontier.astype(dtype=jnp.int32))
        tv_tp1 = total_variation(next_state.map_frontier.astype(dtype=jnp.int32))
        reward_tv_incremental = -(tv_tp1 - tv_t) / (self.v_max * self.tau) * 0.0075

        num_weed = state.map_weed.sum() - next_state.map_weed.sum()
        reward_weed_cut = num_weed * 5.0
        x_t, y_t = state.position.round().astype(jnp.int32)
        x_tp1, y_tp1 = next_state.position.round().astype(jnp.int32)
        if self.use_apf:
            delta_apf = state.observed_weed[y_tp1, x_tp1] - state.observed_weed[y_t, x_t]
        else:
            delta_apf = 0.
        if self.diff_traj:
            delta_traj = -(state.diff_trajectory[y_tp1, x_tp1] - state.diff_trajectory[y_t, x_t]) * 0.125
        else:
            delta_traj = 0.
        reward_weed_approach = delta_apf * 5.0

        reward_traj_cross = -1.5 * next_state.traj_crossed + delta_traj

        reward = (
                reward_const
                + reward_collision
                + reward_coverage
                + reward_tv_incremental
                + reward_weed_cut
                + reward_weed_approach
                + reward_traj_cross
            # + reward_stiff
            # + reward_dynamic
            # + reward_steer
            # + reward_tv_global
        )
        # reward = reward / 10
        return reward

    def state_info(
            self, state: EnvState
    ) -> dict[str, Any]:
        if self.return_info:
            weed_rate = 1 - state.map_weed.sum() / state.weed_count
            coverage_rate = 1 - state.map_frontier.sum() / state.init_map.sum()
            path_length = state.map_trajectory.sum()
            return {
                'weed_rate': weed_rate,
                'coverage_rate': coverage_rate,
                'path_length': path_length,
            }
        else:
            return {}

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
        img = img.repeat(2, axis=0).repeat(2, axis=1)
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
        # new_vision_mask = (
        #                           (lax.broadcast(jnp.arange(0, PastureFunctional.map_width),
        #                                          sizes=[PastureFunctional.map_height]) - x) ** 2
        #                           + (lax.broadcast(jnp.arange(0, PastureFunctional.map_height),
        #                                            sizes=[PastureFunctional.map_width]).swapaxes(0, 1)
        #                              - y) ** 2
        #                   ) <= PastureFunctional.r_vision ** 2
        new_vision_mask = jnp.zeros([
            PastureFunctional.map_height + 2 * PastureFunctional.r_vision,
            PastureFunctional.map_width + 2 * PastureFunctional.r_vision],
            dtype=jnp.bool_)
        if PastureFunctional.triangle_obs:
            rotated_vision_mask = rotate_nearest(
                image=lax.broadcast(PastureFunctional.vision_mask, sizes=[1, ]).transpose(1, 2, 0),
                angle=-(jnp.pi / 2 + state.theta[0]),
                # mode='constant',
            ).transpose(2, 0, 1)[0]
        else:
            rotated_vision_mask = PastureFunctional.vision_mask
        new_vision_mask = lax.dynamic_update_slice(new_vision_mask,
                                                   rotated_vision_mask,
                                                   start_indices=(y, x))
        new_vision_mask = lax.dynamic_slice(new_vision_mask,
                                            start_indices=(PastureFunctional.r_vision, PastureFunctional.r_vision),
                                            slice_sizes=(PastureFunctional.map_height, PastureFunctional.map_width))
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
            lax.broadcast(state.map_trajectory != 0, sizes=[3]).transpose(1, 2, 0),
            jnp.array([0, 128, 255], dtype=jnp.uint8),
            img
        )
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
