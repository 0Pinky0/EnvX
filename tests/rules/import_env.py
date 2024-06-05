import gymnasium as gym
import numpy as np
from gymnasium.wrappers import HumanRendering
import envx.cpp  # noqa

render = True
env = gym.make(
    'Pasture',
    render_mode='rgb_array' if render else None,
    action_type="continuous",
    # weed_count=20,
    # gaussian_weed=True,
    return_map=True,
    # num_obstacle_min=0,
    # num_obstacle_max=0,
)
if render:
    env = HumanRendering(env)
obs, _ = env.reset()
if render:
    env.render()


class Agent:
    def __init__(self):
        pass

    def __call__(self, obs):
        action = [1., 0.]
        return action


agent = Agent()
actions = [[10, -np.pi], [10, 0], [10, 0], [10, 0], [10, 0], [10, 0], [10, 0], [10, 0], [10, 0], [10, 0]]
idx = 0

while True:
    # action = env.action_space.sample()
    # [linear_velocity, angular_velocity]
    action = [2.5, 0.]
    action = actions[idx]
    idx = (idx + 1) % len(actions)
    if idx == 1:
        theta = obs['theta']
        new_theta = theta + action[1] * 0.5
        new_theta = (new_theta + np.pi) % (2 * np.pi) - np.pi
        print(obs['theta'] / np.pi)
        print(new_theta / np.pi)
        print(action)
    # Dict {
    #  'map': array([4, H, W]) 待探索区域， 障碍物， 杂草， 轨迹
    #  'position': array([x, y])
    #  'theta': array([theta])
    # }
    # action = agent(obs)
    obs, reward, done, truncated, _ = env.step(action)
    # print(obs)
    if render:
        env.render()
    if done:
        obs, _ = env.reset()
        if render:
            env.render()
env.close()
