import gymnasium as gym
from gymnasium.wrappers import HumanRendering
import time
import envx.cpp  # noqa

render = False
env = gym.make(
    'Pasture',
    render_mode='rgb_array' if render else None,
    save_pixels=False,
    action_type="discrete",
    rotate_obs=True,
    sgcnn=True,
    global_obs=True,
    use_traj=True,
    use_apf=True,
)
if render:
    env = HumanRendering(env)
obs, _ = env.reset()
if render:
    env.render()
total = 100
i = 0
count = []

while True:
    if i == total:
        break
    action = env.action_space.sample()
    # action = 2 * 1
    tic = time.time()
    obs, reward, done, truncated, _ = env.step(action)
    toc = time.time()
    print(f"{i} / {total} | Elapsed time: {100 * (toc - tic)}ms")
    if i > 0:
        count.append(toc - tic)
    i += 1
    # print(reward)
    if render:
        env.render()
    if done:
        obs, _ = env.reset()
        if render:
            env.render()
env.close()
avg_time = sum(count) / len(count)

print(f'Average time: {avg_time * 100:.4}ms')
