from enum import IntEnum
from typing import Self


class GameState(IntEnum):
    """
    A class representing the current state of the game

    States:
        LOSE = 0
        WIN = 1
        PLAY = 2
    """

    LOSE = 0
    WIN = 1
    PLAY = 2

    def combine_state(self, other: Self) -> Self:
        """Layers states on top of each other, the most important one coming on top"""
        if self < other:
            return self
        else:
            return other
