import gymnasium as gym
from gymnasium.wrappers import HumanRendering
import envx.cpp

# env = gym.make('LawnMowing')
env = gym.make('LawnMowingRender', render_mode='rgb_array')
env = HumanRendering(env)
obs, _ = env.reset()
env.render()

while True:
    action = env.action_space.sample()
    # action = [7, -1]
    obs, reward, done, truncated, _ = env.step(action)
    # if done:
    #     print(obs)
    env.render()
    if done:
        obs, _ = env.reset()
env.close()