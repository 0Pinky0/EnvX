from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import numpy as np

if TYPE_CHECKING:
    import pygame


class FarmlandRenderState(NamedTuple):
    """A named tuple which contains the full render state of the Cliffwalking Env. This is static during the episode."""

    screen: pygame.surface
    shape: tuple[int, int]
    nS: int
    cell_size: tuple[int, int]
    cliff: np.ndarray
    elf_images: tuple[pygame.Surface, pygame.Surface, pygame.Surface, pygame.Surface]
    start_img: pygame.Surface
    goal_img: pygame.Surface
    bg_imgs: tuple[str, str]
    mountain_bg_img: tuple[pygame.Surface, pygame.Surface]
    near_cliff_imgs: tuple[str, str]
    near_cliff_img: tuple[pygame.Surface, pygame.Surface]
    cliff_img: pygame.Surface
