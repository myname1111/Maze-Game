# https://youtu.be/qSBmjSsscI4?si=vqTnT000AKvSXsvN
# Tutorial Pygames : https://youtu.be/AY9MnQ4x3zk?si=MllMZl8EfBFfR9KR
# https://youtu.be/qSBmjSsscI4?si=vqTnT000AKvSXsvN
# Tutorial Pygames : https://youtu.be/AY9MnQ4x3zk?si=MllMZl8EfBFfR9KR
from dataclasses import dataclass
from typing import Optional, Self, Tuple

import pygame

dir = "E:/Python/maze_game_test"


@dataclass
class Vec2:
    x: float
    y: float

    def to_tuple(self) -> Tuple[float, float]:
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


class Player:

    def __init__(self, position: Optional[Vec2] = None) -> None:
        self.position = Vec2(0, 0) if position is None else position
        self.image = pygame.image.load(f"{dir}/imgs/player.png")

    def on_key(self, key: int):
        if key == pygame.K_LEFT:
            self.position.x -= 10
        elif key == pygame.K_RIGHT:
            self.position.x += 10

    def render(self, screen: pygame.Surface):
        screen.blit(self.image, self.position.to_tuple())

def get_points(pos: Vec2, size: Vec2) -> Tuple[Vec2, Vec2, Vec2, Vec2]:
    return (
        Vec2(pos.x, pos.y),
        Vec2(pos.x, pos.y + size.y),
        Vec2(pos.x + size.x, pos.y),
        Vec2(pos.x + size.x, pos.y + size.y),
    )

@dataclass
class Maze:
    maze: list[bool]
    cols: int
    rows: int

    def is_collide(self, row: int, col: int) -> bool:
        return self.maze[row + self.rows * col]


@dataclass
class MazeSprite:
    cell_size: Vec2
    cell_image: pygame.Surface
    maze: Maze
    offset: Vec2

    def to_index(self, pos: Vec2) -> Tuple[int, int]:
        out = (pos - self.offset) // self.cell_size
        return (int(out.x), int(out.y))

    def is_point_collide(self, pos: Vec2) -> bool:
        index = self.to_index(pos)
        return self.maze.is_collide(index[0], index[1])

    def collide_with_sprite(self, other_pos: Vec2, other_size: Vec2) -> bool:
        points = get_points(other_pos, other_size)
        is_collide = False
        for point in points:
            is_collide |= self.is_point_collide(point)
        return is_collide

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

    window.fill((0, 0, 0))
