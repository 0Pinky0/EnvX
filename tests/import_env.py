import gymnasium as gym
from gymnasium.wrappers import HumanRendering
import envx.cpp  # noqa

render = True
env = gym.make('LawnMowing',
               render_mode='rgb_array' if render else None,
               save_pixels=True,
               action_type="continuous",
               rotate_obs=True,
               pbc=False)
if render:
    env = HumanRendering(env)
obs, _ = env.reset()
if render:
    env.render()

while True:
    action = env.action_space.sample()
    # action = [2, 0.1]
    obs, reward, done, truncated, _ = env.step(action)
    print(reward)
    if render:
        env.render()
    if done:
        obs, _ = env.reset()
        if render:
            env.render()
env.close()
