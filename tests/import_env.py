import gymnasium as gym
from gymnasium.wrappers import HumanRendering
import envx.cpp

# env = gym.make('phys2d/CartPole-v0', render_mode='rgb_array')
env = gym.make('LawnMowing', render_mode='rgb_array')
env = HumanRendering(env)
obs, _ = env.reset()
env.render()

while True:
    action = env.action_space.sample()
    obs, reward, done, truncated, _ = env.step(action)
    print(reward)
    env.render()
    if done:
        obs, _ = env.reset()
env.close()