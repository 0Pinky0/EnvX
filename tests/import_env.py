import gymnasium as gym
from gymnasium.wrappers import HumanRendering
import envx.cpp

# env = gym.make('LawnMowing')
env = gym.make('LawnMowing', render_mode='rgb_array', save_pixels=False, continuous=False)
env = HumanRendering(env)
obs, _ = env.reset()
env.render()

while True:
    action = env.action_space.sample()
    # action = [4, 11]
    obs, reward, done, truncated, _ = env.step(action)
    # print(reward)
    # print(obs.keys())
    print(action)
    # if done:
    #     print(obs)
    env.render()
    if done:
        obs, _ = env.reset()
env.close()