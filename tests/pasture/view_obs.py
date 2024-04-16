import numpy as np

import gymnasium as gym
from gymnasium.wrappers import HumanRendering  # noqa
import envx.dcpp  # noqa
from envx.cpp.pasture.pasture import PastureFunctional

import pygame

pygame.init()
pygame.display.init()
r_obs = PastureFunctional.r_obs * 2
scale_up = 3
window = pygame.display.set_mode((r_obs * scale_up, r_obs * scale_up))
clock = pygame.time.Clock()

env = gym.make('Pasture', rotate_obs=True)
state, _ = env.reset()

while True:
    obs = state['observation']
    mask_frontier = np.broadcast_to(obs[0], shape=(3, r_obs, r_obs)).transpose(1, 2, 0)
    mask_obstacle = np.broadcast_to(obs[1], shape=(3, r_obs, r_obs)).transpose(1, 2, 0)
    mask_weed = np.broadcast_to(obs[2], shape=(3, r_obs, r_obs)).transpose(1, 2, 0)
    # mask_self = np.broadcast_to(obs[3], shape=(3, r_obs, r_obs)).transpose(1, 2, 0)
    # mask_vision = np.broadcast_to(obs[4], shape=(3, r_obs, r_obs)).transpose(1, 2, 0)
    mask_traj = np.broadcast_to(obs[3], shape=(3, r_obs, r_obs)).transpose(1, 2, 0)
    img = np.ones([r_obs, r_obs, 3], dtype=np.uint8) * np.array([255, 215, 0])
    img = np.where(np.logical_not(mask_frontier), np.array([255, 255, 255], dtype=np.uint8), img)
    # img = np.where(mask_vision, np.array([64, 64, 64], dtype=np.uint8), img)
    # img = np.where(mask_self, np.array([0, 255, 0], dtype=np.uint8), img)
    img = np.where(mask_obstacle, np.array([0, 0, 255], dtype=np.uint8), img)
    img = np.where(mask_weed, (1 - mask_weed) * np.array(
        [0, 255, 255], dtype=np.uint8
    ) + np.array(
        [255, 0, 0], dtype=np.uint8
    ), img)
    img = np.where(mask_traj, np.array([0, 255, 255], dtype=np.uint8), img)

    img = img.repeat(scale_up, axis=0).repeat(scale_up, axis=1)

    surf = pygame.surfarray.make_surface(img)
    window.blit(surf, (0, 0))
    pygame.event.pump()
    clock.tick(50)
    pygame.display.flip()

    action = env.action_space.sample()
    # action = [2, -0.1]
    # action = [0, 0]
    state, reward, done, truncated, _ = env.step(action)
    print(reward)
    if done:
        state, _ = env.reset()
env.close()
