from dataclasses import dataclass
from typing import Self, Tuple


@dataclass
class Vec2:
    """
    A class representing a vector of 2 numbers

    Attributes:
        x (float): The first number
        y (float): The second number
    """

    x: float
    y: float

    def to_tuple(self) -> Tuple[float, float]:
        """
        A function for a Vec2 into a tuple

        Returns:
            (float, float): The resulting tuple
        """
        return (self.x, self.y)

    def __add__(self, other: Self):
        return Vec2(self.x + other.x, self.y + other.y)

    def __sub__(self, other: Self):
        return Vec2(self.x - other.x, self.y - other.y)

    def __mul__(self, other: Self):
        return Vec2(self.x * other.x, self.y * other.y)

    def __truediv__(self, other: Self):
        return Vec2(self.x / other.x, self.y / other.y)

    def __floordiv__(self, other: Self):
        return Vec2(self.x // other.x, self.y // other.y)

    def __neg__(self):
        return Vec2(-self.x, -self.y)


def vec2_from_int_tuple(tup: Tuple[int | float, int | float]) -> Vec2:
    """
    A function for turning tuples of numbers into a Vec2

    Args:
        tup (int | float, int | float): The input tuple

    Returns:
        Vec2: The outputted vector
    """
    return Vec2(tup[0], tup[1])
