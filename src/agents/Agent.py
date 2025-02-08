from abc import ABC, abstractmethod

from GameState import GameState


class Agent(ABC):
    def __init__(self, name: str):
        self._name = name

    @abstractmethod
    def choose_move(self, game_state: GameState) -> int:
        """
        Abstract method to choose a move based on the game state.
        Must be implemented by subclasses.
        """
        pass

    @property
    def name(self) -> str:
        """Get the name of the agent."""
        return self._name

    @name.setter
    def name(self, value: str):
        """Set the name of the agent."""
        self._name = value