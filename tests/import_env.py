import gymnasium as gym
from gymnasium.wrappers import HumanRendering
import envx.cpp  # noqa

render = True
env = gym.make(
    'PastureToy',
    render_mode='rgb_array' if render else None,
    save_pixels=False,
    action_type="discrete",
    prevent_stiff=True,
    rotate_obs=True,
)
if render:
    env = HumanRendering(env)
obs, _ = env.reset()
if render:
    env.render()

while True:
    action = env.action_space.sample()
    # action = [2, 0.1]
    # action = 0
    # nvec = [4, 19]
    # v_linear = 1
    # v_angular = 9
    # action = (v_angular * 1) + (v_linear * nvec[1])
    obs, reward, done, truncated, _ = env.step(action)
    print(reward)
    if render:
        env.render()
    if done:
        obs, _ = env.reset()
        if render:
            env.render()
env.close()
