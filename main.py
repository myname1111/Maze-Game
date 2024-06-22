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


@dataclass
class Keys:
    is_key_down: dict[int, bool]

    def update(self, event: pygame.event.Event):
        if event.type == pygame.KEYDOWN:
            self.is_key_down[event.key] = True
        elif event.type == pygame.KEYUP:
            self.is_key_down[event.key] = False

    def pressed(self, key: int) -> bool:
        if key in self.is_key_down:
            return self.is_key_down[key]
        else:
            return False


class Maze:
    def __init__(self, in_str: str):
        self.rows = 0
        self.cols = 1
        self.maze = []
        for char in in_str:
            if char == "\n":
                self.rows = 0
                self.cols += 1
            else:
                self.rows += 1

            if char == "#":
                self.maze.append(True)
            elif char == " ":
                self.maze.append(False)

    def is_collide(self, row: int, col: int) -> bool:
        if row >= self.rows:
            return False
        if col >= self.rows:
            return False

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

    def from_index(self, pos: Tuple[int, int]) -> Vec2:
        return Vec2(pos[0], pos[1]) * self.cell_size + self.offset

    def is_point_collide(self, pos: Vec2) -> bool:
        index = self.to_index(pos)
        return self.maze.is_collide(index[0], index[1])

    def collide_with_sprite(self, other_pos: Vec2, other_size: Vec2) -> bool:
        points = get_points(other_pos, other_size)
        is_collide = False
        for point in points:
            is_collide |= self.is_point_collide(point)
        return is_collide

    def render(self, screen: pygame.Surface):
        for idx, cell in enumerate(self.maze.maze):
            x = idx % self.maze.cols
            y = idx // self.maze.cols
            if cell:
                screen.blit(self.cell_image, self.from_index((x, y)).to_tuple())


class Player:

    def __init__(self, position: Optional[Vec2] = None) -> None:
        self.position = Vec2(0, 0) if position is None else position
        self.image = pygame.image.load(f"{dir}/imgs/player.png")
        self.is_alive = True

    def on_key(self, key: Keys):
        if key.pressed(pygame.K_LEFT):
            self.position.x -= 1
        if key.pressed(pygame.K_RIGHT):
            self.position.x += 1
        if key.pressed(pygame.K_DOWN):
            self.position.y += 1
        if key.pressed(pygame.K_UP):
            self.position.y -= 1

    def collision_detection(self, maze: MazeSprite):
        is_collide = maze.collide_with_sprite(self.position, Vec2(64, 64))
        if is_collide:
            self.is_alive = False

    def render(self, screen: pygame.Surface):
        if not self.is_alive:
            return

        screen.blit(self.image, self.position.to_tuple())


def get_points(pos: Vec2, size: Vec2) -> Tuple[Vec2, Vec2, Vec2, Vec2]:
    return (
        Vec2(pos.x, pos.y),
        Vec2(pos.x, pos.y + size.y),
        Vec2(pos.x + size.x, pos.y),
        Vec2(pos.x + size.x, pos.y + size.y),
    )


pygame.init()
window = pygame.display.set_mode((1280, 720))
running = True
player = Player()
keys = Keys({})
maze = MazeSprite(
    Vec2(64, 64),
    pygame.image.load(f"{dir}/imgs/wall.png"),
    Maze(
        "\n".join(
            [
                "   ##   ",
                "   ##   ",
                "#  ##  #",
                "#  ##  #",
                "#  ##  #",
                "#      #",
                "#      #",
                "########",
            ]
        )
    ),
    Vec2(0, 0),
)

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            print("quit game")
            running = False
        keys.update(event)

    player.on_key(keys)

    player.collision_detection(maze)

    player.render(window)
    maze.render(window)

    pygame.display.flip()

    window.fill((0, 0, 0))

    if not player.is_alive:
        print("YOU DIED")
        break
