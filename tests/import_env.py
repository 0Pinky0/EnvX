import gymnasium as gym
from gymnasium.wrappers import HumanRendering
import envx.cpp

env = gym.make('LawnMowing', render_mode='rgb_array', save_pixels=True, action_type="discrete", rotate_obs=True, pbc=True)
env = HumanRendering(env)
obs, _ = env.reset()
env.render()

while True:
    action = env.action_space.sample()
    # action = [4, 11]
    obs, reward, done, truncated, _ = env.step(action)
    print(obs['pixels'].shape)
    env.render()
    if done:
        obs, _ = env.reset()
env.close()