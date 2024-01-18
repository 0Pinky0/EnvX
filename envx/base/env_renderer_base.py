"""Base class and definitions for an alternative, functional backend for gym envs, particularly suitable for hardware accelerated and otherwise transformed environments."""
from __future__ import annotations

from typing import Any, Generic

import numpy as np
from gymnasium.experimental.functional import StateType, RenderStateType


class EnvRendererBase(
    Generic[StateType, RenderStateType]
):
    """Base class (template) for functional envs.

    This API is meant to be used in a stateless manner, with the environment state being passed around explicitly.
    That being said, nothing here prevents users from using the environment statefully, it's just not recommended.
    A functional env consists of the following functions (in this case, instance methods):
    - initial: returns the initial state of the POMDP
    - observation: returns the observation in a given state
    - transition: returns the next state after taking an action in a given state
    - reward: returns the reward for a given (state, action, next_state) tuple
    - terminal: returns whether a given state is terminal
    - state_info: optional, returns a dict of info about a given state
    - step_info: optional, returns a dict of info about a given (state, action, next_state) tuple

    The class-based structure serves the purpose of allowing environment constants to be defined in the class,
    and then using them by name in the code itself.

    For the moment, this is predominantly for internal use. This API is likely to change, but in the future
    we intend to flesh it out and officially expose it to end users.
    """

    def __init__(self, options: dict[str, Any] | None = None):
        """Initialize the environment constants."""
        self.__dict__.update(options or {})

    def render_image(
        self, state: StateType, render_state: RenderStateType
    ) -> tuple[RenderStateType, np.ndarray]:
        """Show the state."""
        raise NotImplementedError

    def render_init(self, **kwargs) -> RenderStateType:
        """Initialize the render state."""
        raise NotImplementedError

    def render_close(self, render_state: RenderStateType):
        """Close the render state."""
        raise NotImplementedError
