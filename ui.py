from typing import Tuple

import pygame

from staging import GameState
from vector import Vec2, vec2_from_int_tuple


class Button:
    """
    A class representing a simple button

    Attributes:
        img (pygame.surface): The image of the button to be used
        pos (Vec2): The position of the button in the screen
        size (Vec2): The size of the button
    """

    def __init__(
        self, img: pygame.Surface, text: pygame.Surface, size: Vec2, dest: Vec2
    ):
        """
        Creates a new button

        Args:
            img (pygame.Surface): The image of the button to be used
            text (pygame.Surface): The image of the text to be used
            size (Vec2): The size of the button
            dest (Vec2): The position of the button in the screen
        """
        self.img = pygame.transform.scale(img, size.to_tuple())
        text_center = text.get_rect(center=(size / Vec2(2, 2)).to_tuple())
        self.img.blit(text, text_center)
        self.pos = dest
        self.size = size

    def render(self, dest: pygame.Surface):
        """
        Render the button

        Args:
            dest (pygame.Surface): Where the buttoon should render to
        """
        dest.blit(self.img, self.pos.to_tuple())

    def is_press(self, mouse_pos: Vec2) -> bool:
        """
        Given a mouse position on he screen, this function checks if that mouse is inside the button

        Args:
            mouse_pos (Vec2): The position of the mouse

        Returns:
            bool: Whether or not the mouse is in the button
        """

        pos2 = self.pos + self.size
        in_x = self.pos.x < mouse_pos.x < pos2.x
        in_y = self.pos.y < mouse_pos.y < pos2.y
        return in_x and in_y


def on_mouse_click(
    game_state: GameState,
    mouse: Tuple[float, float],
    win_button: Button,
    lose_button: Button,
) -> GameState:
    if game_state == GameState.LOSE:
        is_press = lose_button.is_press(vec2_from_int_tuple(mouse))
        if not is_press:
            return GameState.PLAY
        return GameState.LOSE
    elif game_state == GameState.WIN:
        is_press = win_button.is_press(vec2_from_int_tuple(mouse))
        if not is_press:
            return GameState.PLAY
        return GameState.WIN

    return GameState.PLAY
