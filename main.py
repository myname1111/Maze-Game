# https://youtu.be/qSBmjSsscI4?si=vqTnT000AKvSXsvN
# Tutorial Pygames : https://youtu.be/AY9MnQ4x3zk?si=MllMZl8EfBFfR9KR
# https://youtu.be/qSBmjSsscI4?si=vqTnT000AKvSXsvN
# Tutorial Pygames : https://youtu.be/AY9MnQ4x3zk?si=MllMZl8EfBFfR9KR
from dataclasses import dataclass
from typing import Optional, Tuple

import pygame

dir = "E:/Python/maze_game_test"


@dataclass
class Vec2:
    x: int
    y: int

    def to_tuple(self) -> Tuple[int, int]:
        return (self.x, self.y)


class Player:

    def __init__(self, position: Optional[Vec2] = None) -> None:
        self.position = Vec2(0, 0) if position is None else position
        self.image = pygame.image.load(f"{dir}/imgs/player.png")

    def on_key(self, key: int):
        if key == pygame.K_LEFT:
            self.position.x += 10
        elif key == pygame.K_RIGHT:
            self.position.x -= 10

    def render(self, screen: pygame.Surface):
        screen.blit(self.image, self.position.to_tuple())


pygame.init()
window = pygame.display.set_mode((1280, 720))
running = True
player = Player()

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            print("quit game")
            running = False
        if event.type == pygame.KEYDOWN:
            player.on_key(event.key)

    player.render(window)
    pygame.display.flip()
