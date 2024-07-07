import random
from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Optional, Self, Tuple

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


def get_cell_in_grid(paths, index: Tuple[int, int]):
    return paths[index[0]][index[1]]


def move_index_by_direction(index: Tuple[int, int], direction: int) -> Tuple[int, int]:
    # print(index)
    if direction == 0:
        return (index[0], index[1] - 1)
    elif direction == 1:
        return (index[0], index[1] + 1)
    elif direction == 2:
        return (index[0] + 1, index[1])
    elif direction == 3:
        return (index[0] - 1, index[1])
    else:
        assert False


def unwrap[T](input_val: Optional[T]) -> T:
    if input_val is not None:
        return input_val
    else:
        assert False


def is_valid_cell_to_go_to(grid, cell_in_direction: Tuple[int, int]) -> bool:
    if cell_in_direction[0] < 0:
        return False
    if cell_in_direction[1] < 0:
        return False

    try:
        is_connected = get_cell_in_grid(grid, cell_in_direction) is not None
    except IndexError:
        return False

    return is_connected


def get_next_cell_in_grid(
    curr_cell: Tuple[int, int], grid
) -> Tuple[Tuple[int, int], int | None]:
    possible_directions = [0, 1, 2, 3]
    random.shuffle(possible_directions)
    next_cell = None
    new_cell_direction = None
    for direction in possible_directions:
        cell_in_direction = move_index_by_direction(curr_cell, direction)

        if is_valid_cell_to_go_to(grid, cell_in_direction):
            next_cell = cell_in_direction
            new_cell_direction = direction
            break

    if next_cell is not None:
        return (next_cell, new_cell_direction)

    dir_towards_prev_cell: Optional[int] = get_cell_in_grid(grid, curr_cell)
    dir_towards_prev_cell = unwrap(dir_towards_prev_cell)
    prev_cell = move_index_by_direction(curr_cell, dir_towards_prev_cell)

    return (prev_cell, new_cell_direction)


def get_opposite_direction(dir: int) -> int:
    if dir == 0:
        return 1
    elif dir == 1:
        return 0
    elif dir == 2:
        return 3
    elif dir == 3:
        return 2
    else:
        assert False


class Maze:
    def __init__(self, size: Tuple[int, int]):
        self.cols = size[0]
        self.rows = size[1]
        self.maze = ""
        # Each cell represents a direction to the cell before them
        # Any is not preferable, maybe a bug with linter?
        paths: list[list[Any]] = [[[None] * size[0]] * size[1]][0]
        print(paths)
        cells = size[0] * size[1]
        reached = 1
        curr_cell = (0, 0)
        while reached < cells:
            # 0: DOWN
            # 1: UP
            # 2: RIGHT
            # 3: LEFT
            print(curr_cell, "curr_cell")
            (next_cell, new_cell_direction) = get_next_cell_in_grid(curr_cell, paths)
            # print(next_cell)
            is_reached_new_cell = new_cell_direction is not None
            curr_cell = next_cell
            if is_reached_new_cell:
                reached += 1
                print(reached, "reached")
                direction_towards_prev_cell = get_opposite_direction(new_cell_direction)
                paths[curr_cell[0]][curr_cell[1]] = direction_towards_prev_cell
                # TODO: Set the direction of a cell to the cell it was from if it is reached
        print(paths)

    def get(self, col: int, row: int) -> str:
        if row >= self.rows:
            return " "
        if col >= self.cols:
            return " "

        return self.maze[col + self.cols * row]


class MazeSprite:
    def __init__(
        self,
        cell_size: Vec2,
        cell_image: dict[str, pygame.Surface],
        maze: Maze,
        offset: Vec2,
    ) -> None:
        self.cell_size = cell_size
        self.cell_image = {
            cell: pygame.transform.scale(cell_image[cell], self.cell_size.to_tuple())
            for cell in cell_image
        }
        self.maze = maze
        self.offset = offset

    def to_index(self, pos: Vec2) -> Tuple[int, int]:
        out = (pos - self.offset) // self.cell_size
        return (int(out.x), int(out.y))

    def from_index(self, pos: Tuple[int, int]) -> Vec2:
        return Vec2(pos[0], pos[1]) * self.cell_size + self.offset

    def collide_with_sprite(self, other_pos: Vec2, other_size: Vec2) -> list[str]:
        points = get_bounding_box(other_pos, other_size)
        grid_index_bounding_box = [self.to_index(point) for point in points]
        indicies = get_list_of_indicies_inside_grid_index_bounding_box(
            grid_index_bounding_box[0], grid_index_bounding_box[1]
        )
        return [self.maze.get(index[0], index[1]) for index in indicies]

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

    def __init__(
        self,
        position: Optional[Vec2] = None,
        size: Optional[Vec2] = None,
        speed: float = 1,
    ) -> None:
        self.position = Vec2(0, 0) if position is None else position
        self.image = pygame.image.load(f"{dir}/imgs/player.png")
        if size is not None:
            self.image = pygame.transform.scale(self.image, size.to_tuple())
        self.speed = speed

    def on_key(self, key: Keys, state: GameState):
        if state == GameState.LOSE:
            return

        if key.pressed(pygame.K_LEFT):
            self.position.x -= self.speed
        if key.pressed(pygame.K_RIGHT):
            self.position.x += self.speed
        if key.pressed(pygame.K_DOWN):
            self.position.y += self.speed
        if key.pressed(pygame.K_UP):
            self.position.y -= self.speed

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
        collided_cells = maze.collide_with_sprite(
            self.position, vec2_from_int_tuple(self.image.get_size())
        )
        state = init_state
        for collided_cell in collided_cells:
            state = state.combine_state(self.collide_with_cell(collided_cell, state))
        return state

    def render(self, screen: pygame.Surface, state: GameState):
        if state == GameState.LOSE:
            return

        screen.blit(self.image, self.position.to_tuple())


def get_list_of_indicies_inside_grid_index_bounding_box(
    start: Tuple[int, int], to: Tuple[int, int]
) -> list[Tuple[int, int]]:
    return [
        (x, y) for y in range(start[1], to[1] + 1) for x in range(start[0], to[0] + 1)
    ]


def get_bounding_box(pos: Vec2, size: Vec2) -> Tuple[Vec2, Vec2]:
    return (
        Vec2(pos.x, pos.y),
        Vec2(pos.x + size.x, pos.y + size.y),
    )


pygame.init()
window = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
running = True
player = Player(speed=0.5, size=Vec2(32, 32))
keys = Keys({})
maze = MazeSprite(
    Vec2(64, 64),
    {
        "#": pygame.image.load(f"{dir}/imgs/wall.png"),
        "X": pygame.image.load(f"{dir}/imgs/win.png"),
    },
    Maze((8, 8)),
    Vec2(0, 0),
)

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
