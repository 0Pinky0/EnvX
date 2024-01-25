import gymnasium as gym
from gymnasium.wrappers import HumanRendering
import envx.cpp

env = gym.make('LawnMowing')
env = gym.make('LawnMowing', render_mode='rgb_array')
env = HumanRendering(env)
obs, _ = env.reset()
env.render()
# count = 0

while True:
    # if count == 1000:
    #     break
    # count += 1
    action = env.action_space.sample()
    obs, reward, done, truncated, _ = env.step(action)
    # print(type(obs))
    env.render()
    if done:
        obs, _ = env.reset()
env.close()