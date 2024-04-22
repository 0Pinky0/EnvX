import gymnasium as gym
from gymnasium.wrappers import HumanRendering
import envx.cpp  # noqa
import numpy as np

render = True
env = gym.make(
    'LawnMowing',
    render_mode='rgb_array' if render else None,
    save_pixels=False,
    action_type="continuous",
    # action_type="discrete",
    prevent_stiff=True,
    rotate_obs=True,
)
if render:
    env = HumanRendering(env)
obs, _ = env.reset()
if render:
    env.render()
step = 0
reward = 10
status = 0

while True:
    # action = [2, 0.1]
    # action = 0
    # nvec = [4, 19]
    # v_linear = 1
    # v_angular = 9
    # action = (v_angular * 1) + (v_linear * nvec[1])
    # action = env.action_space.sample()
    if reward <= -0.01 or status:
        match status:
            case 2:
                action = [5, 0]
                status = 0
            case _:
                action = [0, -np.pi]
                status += 1
    else:
        action = [5, 0]
    print(action, status)
    obs, reward, done, truncated, _ = env.step(action)
    # print(reward)
    print(f'step {step} / 2000: {reward}')
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