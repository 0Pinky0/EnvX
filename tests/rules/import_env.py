import gymnasium as gym
from gymnasium.wrappers import HumanRendering
import envx.cpp  # noqa

render = True
env = gym.make(
    'Pasture',
    render_mode='rgb_array' if render else None,
    action_type="continuous",
    weed_count=600,
    gaussian_weed=True,
    return_map=True,
    num_obstacle_min=0,
    num_obstacle_max=0,
)
if render:
    env = HumanRendering(env)
obs, _ = env.reset()
if render:
    env.render()

while True:
    # action = env.action_space.sample()
    action = [2, 0.]
    obs, reward, done, truncated, _ = env.step(action)
    print(obs)
    if render:
        env.render()
    if done:
        obs, _ = env.reset()
        if render:
            env.render()
env.close()
