import gymnasium as gym
from gymnasium.wrappers import HumanRendering
import envx.cpp  # noqa
import numpy as np

render = True
env = gym.make(
    'Pasture',
    render_mode='rgb_array' if render else None,
    save_pixels=False,
    action_type="continuous",
    # action_type="discrete",
    prevent_stiff=True,
    rotate_obs=True,
    sgcnn=True,
    use_apf=False,
    weed_count=600,
)
if render:
    env = HumanRendering(env)
obs, _ = env.reset()
if render:
    env.render()
step = 0

while True:
    # action = [2, 0.1]
    # action = 0
    # nvec = [4, 19]
    # v_linear = 1
    # v_angular = 9
    # action = (v_angular * 1) + (v_linear * nvec[1])
    action = env.action_space.sample()
    # action = [5, 0]
    obs, reward, done, truncated, info = env.step(action)
    # print(reward)
    print(f'step {step} / 2000: {reward}')
    print(info)
    step += 1
    if render:
        env.render()
    if done:
        obs, _ = env.reset()
        step = 0
        if render:
            env.render()
        reward = 10
        status = 0
env.close()
