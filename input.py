from dataclasses import dataclass

import pygame


@dataclass
class Keys:
    """
    A class representing a simple map keys and whether or not it is being pressed

    Attributes:
        is_key_down ({int, bool}): The inner dictionary
    """

    is_key_down: dict[int, bool]

    def update(self, event: pygame.event.Event):
        """Updates it with a key update"""

        if event.type == pygame.KEYDOWN:
            self.is_key_down[event.key] = True
        elif event.type == pygame.KEYUP:
            self.is_key_down[event.key] = False

    def pressed(self, key: int) -> bool:
        """Check whether or not a key is pressed"""
        if key in self.is_key_down:
            return self.is_key_down[key]
        else:
            return False
