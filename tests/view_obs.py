import numpy as np

import gymnasium as gym
from gymnasium.wrappers import HumanRendering
import envx.cpp

import pygame

pygame.init()
pygame.display.init()
r_obs = 128 * 2
window = pygame.display.set_mode((r_obs, r_obs))
clock = pygame.time.Clock()

# env = gym.make('LawnMowing')
env = gym.make('LawnMowing', render_mode='rgb_array')
env = HumanRendering(env)
obs, _ = env.reset()
env.render()

while True:
    # action = env.action_space.sample()
    action = [7, 0]
    # action = [0, 0]
    obs, reward, done, truncated, _ = env.step(action)

    mask_frontier = np.broadcast_to(obs[0], shape=(3, r_obs, r_obs)).transpose(1, 2, 0)
    mask_obstacle = np.broadcast_to(obs[1], shape=(3, r_obs, r_obs)).transpose(1, 2, 0)
    # img = np.broadcast_to(obs[1], shape=(3, r_obs, r_obs))
    img = np.ones([r_obs, r_obs, 3], dtype=np.uint8) * np.array([65, 227, 72])
    img = np.where(mask_frontier, np.array([255, 255, 255], dtype=np.uint8), img)
    img = np.where(mask_obstacle, np.array([0, 0, 255], dtype=np.uint8), img)

    surf = pygame.surfarray.make_surface(img)
    window.blit(surf, (0, 0))
    pygame.event.pump()
    clock.tick(50)
    pygame.display.flip()
    # if done:
    #     print(obs)
    #     print(reward)
    env.render()
    if done:
        obs, _ = env.reset()
env.close()
