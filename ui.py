from typing import Tuple

import pygame

from staging import GameState
from vector import Vec2, vec2_from_int_tuple


class Button:
    def __init__(
        self, img: pygame.Surface, text: pygame.Surface, size: Vec2, dest: Vec2
    ):
        self.img = pygame.transform.scale(img, size.to_tuple())
        text_center = text.get_rect(center=(size / Vec2(2, 2)).to_tuple())
        self.img.blit(text, text_center)
        self.pos = dest
        self.size = size

    def render(self, dest: pygame.Surface):
        dest.blit(self.img, self.pos.to_tuple())

    def is_press(self, mouse_pos: Vec2) -> bool:
        pos2 = self.pos + self.size
        in_x = self.pos.x < mouse_pos.x < pos2.x
        in_y = self.pos.y < mouse_pos.y < pos2.y
        return in_x and in_y


def on_mouse_click(
    game_state: GameState,
    mouse: Tuple[int, int],
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
