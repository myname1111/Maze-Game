# https://youtu.be/qSBmjSsscI4?si=vqTnT000AKvSXsvN
# Tutorial Pygames : https://youtu.be/AY9MnQ4x3zk?si=MllMZl8EfBFfR9KR
# https://youtu.be/qSBmjSsscI4?si=vqTnT000AKvSXsvN
# Tutorial Pygames : https://youtu.be/AY9MnQ4x3zk?si=MllMZl8EfBFfR9KR
from dataclasses import dataclass
from enum import Enum, IntEnum
from typing import Optional, Self, Tuple

import pygame
from pygame import font

dir = "E:/Python/maze_game_test"
SCREEN_WIDTH = 1280
SCREEN_HEIGHT = 720


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


def vec2_from_int_tuple(tup: Tuple[int | float, int | float]) -> Vec2:
    return Vec2(tup[0], tup[1])


SCREEN_SIZE = Vec2(SCREEN_WIDTH, SCREEN_HEIGHT)


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
    def __init__(self, in_str: list[str]):
        self.cols = len(in_str[0])
        self.rows = len(in_str)
        self.maze = "".join(in_str)

    def get(self, col: int, row: int) -> str:
        if row >= self.rows:
            return " "
        if col >= self.cols:
            return " "

        return self.maze[col + self.cols * row]


@dataclass
class MazeSprite:
    cell_size: Vec2
    cell_image: dict[str, pygame.Surface]
    maze: Maze
    offset: Vec2

    def to_index(self, pos: Vec2) -> Tuple[int, int]:
        out = (pos - self.offset) // self.cell_size
        return (int(out.x), int(out.y))

    def from_index(self, pos: Tuple[int, int]) -> Vec2:
        return Vec2(pos[0], pos[1]) * self.cell_size + self.offset

    def is_point_collide(self, pos: Vec2) -> str:
        index = self.to_index(pos)
        return self.maze.get(index[0], index[1])

    def collide_with_sprite(self, other_pos: Vec2, other_size: Vec2) -> list[str]:
        points = get_points(other_pos, other_size)
        return [self.is_point_collide(point) for point in points]

    def render(self, screen: pygame.Surface):
        for idx, cell in enumerate(self.maze.maze):
            x = idx % self.maze.cols
            y = idx // self.maze.cols

            if cell in self.cell_image:
                screen.blit(self.cell_image[cell], self.from_index((x, y)).to_tuple())


class GameState(IntEnum):
    LOSE = 0
    WIN = 1
    PLAY = 2

    def combine_state(self, other: Self) -> Self:
        if self < other:
            return self
        else:
            return other


class Player:

    def __init__(self, position: Optional[Vec2] = None) -> None:
        self.position = Vec2(0, 0) if position is None else position
        self.image = pygame.image.load(f"{dir}/imgs/player.png")

    def on_key(self, key: Keys, state: GameState):
        if state == GameState.LOSE:
            return

        if key.pressed(pygame.K_LEFT):
            self.position.x -= 1
        if key.pressed(pygame.K_RIGHT):
            self.position.x += 1
        if key.pressed(pygame.K_DOWN):
            self.position.y += 1
        if key.pressed(pygame.K_UP):
            self.position.y -= 1

    def collide_with_cell(self, cell: str, init_state: GameState) -> GameState:
        match cell:
            case " ":
                return init_state
            case "#":
                return GameState.LOSE
            case "X":
                return GameState.WIN
            case _:
                assert False

    def collision_detection(self, maze: MazeSprite, init_state: GameState) -> GameState:
        collided_cells = maze.collide_with_sprite(self.position, Vec2(64, 64))
        state = init_state
        for collided_cell in collided_cells:
            state = state.combine_state(self.collide_with_cell(collided_cell, state))
        print(state)
        return state

    def render(self, screen: pygame.Surface, state: GameState):
        if state == GameState.LOSE:
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
window = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
running = True
player = Player()
keys = Keys({})
maze = MazeSprite(
    Vec2(64, 64),
    {
        "#": pygame.image.load(f"{dir}/imgs/wall.png"),
        "X": pygame.image.load(f"{dir}/imgs/win.png"),
    },
    Maze(
        [
            "   #############",
            "   #        #  #",
            "#  #       X#  #",
            "#  #  ####  #  #",
            "#     #  #     #",
            "#     #  #     #",
            "#######  ####  #",
            "#              #",
            "#              #",
            "#  #  ##########",
            "#  #  #        #",
            "#  ####        #",
            "#        ####   ",
            "#        #  #  X",
            "##########  ####",
        ]
    ),
    Vec2(0, 0),
)

print(maze.maze.rows, maze.maze.cols)

font.init()
font = font.Font(None, 70)
lose = font.render("LOSE", True, (241, 40, 12))
win = font.render("WIN", True, (19, 232, 51))
game_state = GameState.PLAY

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            print("quit game")
            running = False
        keys.update(event)

    player.on_key(keys, game_state)
    player.render(window, game_state)
    maze.render(window)

    game_state = player.collision_detection(maze, game_state)

    if game_state == GameState.LOSE:
        window.blit(lose, lose.get_rect(center=(SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2)))
    if game_state == GameState.WIN:
        window.blit(win, win.get_rect(center=(SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2)))

    pygame.display.flip()

    window.fill((0, 0, 0))
