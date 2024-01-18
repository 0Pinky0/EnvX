from __future__ import annotations

from os import path
from typing import TYPE_CHECKING

import jax
import numpy as np
from gymnasium.error import DependencyNotInstalled
from gymnasium.experimental.functional import StateType

from envx.base.env_renderer_base import EnvRendererBase
from envx.farmland.render.Farmland_render_state import FarmlandRenderState

if TYPE_CHECKING:
    import pygame


class FarmlandRenderer(
    EnvRendererBase[jax.Array, FarmlandRenderState]
):

    def render_init(
            self, screen_width: int = 600, screen_height: int = 500
    ) -> FarmlandRenderState:
        """Returns an initial render state."""
        try:
            import pygame
        except ImportError:
            raise DependencyNotInstalled(
                "pygame is not installed, run `pip install gymnasium[classic_control]`"
            )

        cell_size = (60, 60)
        window_size = (
            4 * cell_size[0],
            12 * cell_size[1],
        )

        pygame.init()
        screen = pygame.Surface((window_size[1], window_size[0]))

        shape = (4, 12)
        nS = 4 * 12
        # Cliff Location
        cliff = np.zeros(shape, dtype=bool)
        cliff[3, 1:-1] = True

        hikers = [
            path.join(path.dirname(__file__), "../toy_text/img/elf_up.png"),
            path.join(path.dirname(__file__), "../toy_text/img/elf_right.png"),
            path.join(path.dirname(__file__), "../toy_text/img/elf_down.png"),
            path.join(path.dirname(__file__), "../toy_text/img/elf_left.png"),
        ]

        cell_size = (60, 60)

        elf_images = [
            pygame.transform.scale(pygame.image.load(f_name), cell_size)
            for f_name in hikers
        ]
        file_name = path.join(path.dirname(__file__), "../toy_text/img/stool.png")
        start_img = pygame.transform.scale(pygame.image.load(file_name), cell_size)
        file_name = path.join(path.dirname(__file__), "../toy_text/img/cookie.png")
        goal_img = pygame.transform.scale(pygame.image.load(file_name), cell_size)
        bg_imgs = [
            path.join(path.dirname(__file__), "../toy_text/img/mountain_bg1.png"),
            path.join(path.dirname(__file__), "../toy_text/img/mountain_bg2.png"),
        ]
        mountain_bg_img = [
            pygame.transform.scale(pygame.image.load(f_name), cell_size)
            for f_name in bg_imgs
        ]
        near_cliff_imgs = [
            path.join(
                path.dirname(__file__), "../toy_text/img/mountain_near-cliff1.png"
            ),
            path.join(
                path.dirname(__file__), "../toy_text/img/mountain_near-cliff2.png"
            ),
        ]
        near_cliff_img = [
            pygame.transform.scale(pygame.image.load(f_name), cell_size)
            for f_name in near_cliff_imgs
        ]
        file_name = path.join(
            path.dirname(__file__), "../toy_text/img/mountain_cliff.png"
        )
        cliff_img = pygame.transform.scale(pygame.image.load(file_name), cell_size)

        return FarmlandRenderState(
            screen=screen,
            shape=shape,
            nS=nS,
            cell_size=cell_size,
            cliff=cliff,
            elf_images=tuple(elf_images),
            start_img=start_img,
            goal_img=goal_img,
            bg_imgs=tuple(bg_imgs),
            mountain_bg_img=tuple(mountain_bg_img),
            near_cliff_imgs=tuple(near_cliff_imgs),
            near_cliff_img=tuple(near_cliff_img),
            cliff_img=cliff_img,
        )

    def render_image(
            self,
            state: StateType,
            render_state: FarmlandRenderState,
    ) -> tuple[FarmlandRenderState, np.ndarray]:
        """Renders an image from a state."""
        try:
            import pygame
        except ImportError:
            raise DependencyNotInstalled(
                "pygame is not installed, run `pip install gymnasium[toy_text]`"
            )
        (
            window_surface,
            shape,
            nS,
            cell_size,
            cliff,
            elf_images,
            start_img,
            goal_img,
            bg_imgs,
            mountain_bg_img,
            near_cliff_imgs,
            near_cliff_img,
            cliff_img,
        ) = render_state

        for s in range(nS):
            row, col = np.unravel_index(s, shape)
            pos = (col * cell_size[0], row * cell_size[1])
            check_board_mask = row % 2 ^ col % 2
            window_surface.blit(mountain_bg_img[check_board_mask], pos)

            if cliff[row, col]:
                window_surface.blit(cliff_img, pos)
            if row < shape[0] - 1 and cliff[row + 1, col]:
                window_surface.blit(near_cliff_img[check_board_mask], pos)
            if s == 36:
                window_surface.blit(start_img, pos)
            if s == nS - 1:
                window_surface.blit(goal_img, pos)
            if s == state.player_position[0] * 12 + state.player_position[1]:
                elf_pos = (pos[0], pos[1] - 0.1 * cell_size[1])
                last_action = state.last_action if state.last_action != -1 else 2
                window_surface.blit(elf_images[last_action], elf_pos)

        return render_state, np.transpose(
            np.array(pygame.surfarray.pixels3d(window_surface)), axes=(1, 0, 2)
        )

    def render_close(self, render_state: FarmlandRenderState) -> None:
        """Closes the render state."""
        try:
            import pygame
        except ImportError as e:
            raise DependencyNotInstalled(
                "pygame is not installed, run `pip install gymnasium[toy-text]`"
            ) from e
        pygame.display.quit()
        pygame.quit()
