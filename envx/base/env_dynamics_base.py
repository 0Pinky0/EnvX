"""Base class and definitions for an alternative, functional backend for gym envs, particularly suitable for hardware accelerated and otherwise transformed environments."""
from __future__ import annotations

from typing import Any, Generic

from gymnasium.experimental.functional import StateType, ObsType, ActType, RewardType, TerminalType


class EnvDynamicsBase(
    Generic[StateType, ObsType, ActType, RewardType, TerminalType]
):
    """Base class (template) for functional envs.
    """

    def __init__(self, options: dict[str, Any] | None = None):
        """Initialize the environment constants."""
        self.__dict__.update(options or {})

    def initial(self, rng: Any) -> StateType:
        """Initial state."""
        raise NotImplementedError

    def transition(self, state: StateType, action: ActType, rng: Any) -> StateType:
        """Transition."""
        raise NotImplementedError

    def observation(self, state: StateType) -> ObsType:
        """Observation."""
        raise NotImplementedError

    def reward(
        self, state: StateType, action: ActType, next_state: StateType
    ) -> RewardType:
        """Reward."""
        raise NotImplementedError

    def terminal(self, state: StateType) -> TerminalType:
        """Terminal state."""
        raise NotImplementedError

    def state_info(self, state: StateType) -> dict:
        """Info dict about a single state."""
        return {}

    def step_info(
        self, state: StateType, action: ActType, next_state: StateType
    ) -> dict:
        """Info dict about a full transition."""
        return {}
