import random
from dataclasses import dataclass
from enum import IntEnum
from typing import Optional, Self, Tuple

import pygame
from pygame import font

dir = "E:/Python/maze_game_test"
SCREEN_WIDTH = 1280
SCREEN_HEIGHT = 720

# 0: DOWN
# 1: UP
# 2: RIGHT
# 3: LEFT

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

    def __neg__(self):
        return Vec2(-self.x, -self.y)


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


def get_cell_in_grid(paths: list[list[Optional[int]]], index: Tuple[int, int]):
    return paths[index[1]][index[0]]


def move_position_by_direction(index: Tuple[int, int], direction: int) -> Tuple[int, int]:
    # print(index)
    if direction == 0:
        return (index[0], index[1] - 1)
    elif direction == 1:
        return (index[0], index[1] + 1)
    elif direction == 2:
        return (index[0] - 1, index[1])
    elif direction == 3:
        return (index[0] + 1, index[1])
    else:
        assert False


def unwrap[T](input_val: Optional[T]) -> T:
    if input_val is not None:
        return input_val
    else:
        assert False


def is_valid_cell_to_go_to(
    grid: list[list[Optional[int]]], cell_in_direction: Tuple[int, int]
) -> bool:
    # If out of range
    if cell_in_direction[0] < 0:
        return False
    if cell_in_direction[1] < 0:
        return False

    is_root = cell_in_direction[0] == 0 and cell_in_direction[1] == 0
    if is_root:
        return False

    try:
        is_connected = get_cell_in_grid(grid, cell_in_direction) is not None
    except IndexError:
        return False

    return not is_connected


def get_next_cell_in_grid(
    curr_cell: Tuple[int, int], grid: list[list[Optional[int]]]
) -> Tuple[Tuple[int, int], int | None]:
    possible_directions = [0, 1, 2, 3]
    random.shuffle(possible_directions)
    # print(possible_directions, "possible_directions")
    next_cell = None
    new_cell_direction = None
    for direction in possible_directions:
        cell_in_direction = move_position_by_direction(curr_cell, direction)

        if is_valid_cell_to_go_to(grid, cell_in_direction):
            next_cell = cell_in_direction
            new_cell_direction = direction
            break

    if next_cell is not None:
        # In this case there is a next cell to go to
        return (next_cell, new_cell_direction)

    # Else, we go to the previous cell
    dir_towards_prev_cell: Optional[int] = get_cell_in_grid(grid, curr_cell)
    dir_towards_prev_cell = get_opposite_direction(unwrap(dir_towards_prev_cell))
    prev_cell = move_position_by_direction(curr_cell, dir_towards_prev_cell)

    return (prev_cell, None)


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


def init_grid(x: int, y: int) -> list[list[Optional[int]]]:
    out = []
    for _ in range(0, y):
        row: list[Optional[int]] = []
        for _ in range(0, x):
            row.append(None)
        out.append(row)
    return out


def create_paths(size: Tuple[int, int]) -> list[list[Optional[int]]]:
    # Each cell represents a direction to the cell before them
    paths = init_grid(size[0], size[1])
    reached = 0
    curr_cell = (0, 0)
    while True:
        new = get_next_cell_in_grid(curr_cell, paths)
        (next_cell, new_cell_direction) = new
        is_reached_new_cell = new_cell_direction is not None
        curr_cell = next_cell
        if is_reached_new_cell:
            reached += 1
            paths[curr_cell[1]][curr_cell[0]] = new_cell_direction
        if curr_cell == (0, 0):
            break

    return paths


def init_maze(size: Tuple[int, int]) -> list[list[str]]:
    maze = []
    for x in range(size[0] * 2 + 1):
        row = []
        for y in range(size[1] * 2 + 1):
            row.append(" " if x % 2 == 1 and y % 2 == 1 else "#")
        maze.append(row)
    return maze

def move_grid_index_by_direction(index: Tuple[int, int], direction: int) -> Tuple[int, int]:
    # print(index)
    if direction == 0:
        return (index[0], index[1] + 1)
    elif direction == 1:
        return (index[0], index[1] - 1)
    elif direction == 2:
        return (index[0] + 1, index[1])
    elif direction == 3:
        return (index[0] - 1, index[1])
    else:
        assert False

def create_walls_from_paths(
    size: Tuple[int, int], paths: list[list[Optional[int]]], win: str = "X"
) -> str:
    maze = init_maze(size)
    for y, row in enumerate(paths):
        for x, cell in enumerate(row):
            if cell is None:
                continue

            (x_new, y_new) = move_grid_index_by_direction((x * 2 + 1, y * 2 + 1), cell)
            
            maze[y_new][x_new] = ' '


    maze[1][size[0] * 2] = win
    return "".join(["".join(row) for row in maze])

def step_back(pos: Tuple[int, int], paths: list[list[Optional[int]]]) -> Tuple[Tuple[int, int], int] | None:
    direction = paths[pos[1]][pos[0]]
    if direction is None:
        return None

    new = move_grid_index_by_direction(pos, direction)
    return (new, direction)

def many_step_back(pos: Tuple[int, int], paths: list[list[Optional[int]]], count: int) -> Tuple[Tuple[int, int], list[int]]:
    curr_pos = pos
    out = []
    for _ in range(count):
        new = step_back(curr_pos, paths)
        if new is None:
            return (curr_pos, out)
        curr_pos = new[0]
        out.append(new[1])
    return (curr_pos, out)

def get_depth(paths: list[list[Optional[int]]], size: Tuple[int, int]) -> list[list[int]]:
    out = []
    for (y, row) in enumerate(paths):
        out_row = []
        for (x, _) in enumerate(row):
            out_row.append(len(many_step_back((x, y), paths, size[0] * size[1])[1]))
        out.append(out_row)
    return out


class Maze:
    def __init__(self, size: Tuple[int, int]):
        self.cols = size[0] * 2 + 1
        self.rows = size[1] * 2 + 1
        self.paths = create_paths(size)
        self.depth = get_depth(self.paths, size)
        print(self.pathfind((2, 0), (2, 1)))
        self.maze = create_walls_from_paths(size, self.paths)

    def pathfind(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> list[int]:
        out = []
        depth1 = self.depth[pos1[1]][pos1[0]]
        depth2 = self.depth[pos2[1]][pos2[0]]
        min_depth = min(depth1, depth2)
        curr_pos1 = pos1
        curr_pos2 = pos2
        directions_to_1 = []
        directions_to_2 = []
        if depth1 > min_depth:
            diff = depth1 - min_depth
            (curr_pos1, directions_to_1) = many_step_back(pos1, self.paths, diff)
        if depth2 > min_depth:
            diff = depth2 - min_depth
            (curr_pos2, directions_to_2) = many_step_back(pos2, self.paths, diff)

        while curr_pos1 != curr_pos2:
            new1 = step_back(curr_pos1, self.paths)
            if new1 is not None:
                curr_pos1 = new1[0]
                directions_to_1.append(new1[1])
            
            new2 = step_back(curr_pos2, self.paths)
            if new2 is not None:
                curr_pos2 = new2[0]
                directions_to_2.append(new2[1])

        towards_lca = list(reversed(directions_to_1))
        from_lca = [get_opposite_direction(dir) for dir in directions_to_2]
        out = from_lca + towards_lca

        return out
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
    
    def to_path_index(self, pos: Vec2) -> Tuple[int, int]:
        sprite_grid_pos = self.to_index(pos)
        return (sprite_grid_pos[0] // 2, sprite_grid_pos[1] // 2)

    def collide_with_sprite(self, other_pos: Vec2, other_size: Vec2) -> list[str]:
        points = get_bounding_box(other_pos, other_size)
        grid_index_bounding_box = [self.to_index(point) for point in points]
        indicies = get_list_of_indicies_inside_grid_index_bounding_box(
            grid_index_bounding_box[0], grid_index_bounding_box[1]
        )
        return [self.maze.get(index[0], index[1]) for index in indicies]

    def render(self, screen: pygame.Surface, offset: Vec2):
        for idx, cell in enumerate(self.maze.maze):
            x = idx % self.maze.cols
            y = idx // self.maze.cols

            if cell in self.cell_image:
                screen.blit(self.cell_image[cell], (self.from_index((x, y)) + offset).to_tuple())


class GameState(IntEnum):
    LOSE = 0
    WIN = 1
    PLAY = 2

    def combine_state(self, other: Self) -> Self:
        if self < other:
            return self
        else:
            return other

def get_direction_from_offset(offset: Tuple[int, int]) -> Optional[int]:
    if offset == (0, 1):
        return 0
    elif offset == (0, -1):
        return 1
    elif offset == (1, 0):
        return 2
    elif offset == (-1, 0):
        return 3

class Player:

    def __init__(
        self,
        maze: MazeSprite,
        position: Optional[Vec2] = None,
        size: Optional[Vec2] = None,
        speed: float = 1,
    ) -> None:
        self.position = Vec2(0, 0) if position is None else position
        self.image = pygame.image.load(f"{dir}/imgs/player.png")
        if size is not None:
            self.image = pygame.transform.scale(self.image, size.to_tuple())
        self.size = self.image.get_size()
        self.speed = speed
        self.path_grid_pos = maze.to_path_index(self.position)

    def on_key(self, key: Keys, state: GameState, maze: MazeSprite) -> Optional[int]:
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

        new_path_grid_pos = maze.to_path_index(self.position * Vec2(2, 2))
        if new_path_grid_pos == self.path_grid_pos:
            return None

        offset = (new_path_grid_pos[0] - self.path_grid_pos[0], new_path_grid_pos[1] - self.path_grid_pos[1])
        direction = get_direction_from_offset(offset)

        self.path_grid_pos = new_path_grid_pos

        return direction

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

        screen.blit(self.image, (SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2))

    def get_bounding_box(self) -> Tuple[Vec2, Vec2]:
        return get_bounding_box(self.position, vec2_from_int_tuple(self.size))


def get_list_of_indicies_inside_grid_index_bounding_box(
    start: Tuple[int, int], to: Tuple[int, int]
) -> list[Tuple[int, int]]:
    return [
        (x, y) for y in range(start[1], to[1] + 1) for x in range(start[0], to[0] + 1)
    ]


def get_bounding_box(pos: Vec2, size: Vec2) -> Tuple[Vec2, Vec2]:
    return (
        pos,
        pos + size
    )

def push_to_direction_list(direction_list: list[int], direction: int) -> list[int]:
    if len(direction_list) == 0:
        return [direction]

    out = direction_list
    opposite = get_opposite_direction(direction)
    if direction_list[0] == opposite:
        return direction_list[1:]
    return [direction] + out

class Enemy:
    def __init__(
        self,
        distance: int,
        maze: MazeSprite,
        position: Optional[Vec2] = None,
        size: Optional[Vec2] = None,
        speed: float = 1,
        moves: Optional[list[int]] = None):
        self.position = Vec2(0, 0) if position is None else position
        self.image = pygame.image.load(f"{dir}/imgs/enemy.png")
        if size is not None:
            self.image = pygame.transform.scale(self.image, size.to_tuple())
        self.size = self.image.get_size()
        self.speed = speed
        self.moves = [] if moves is None else moves
        self.moved_distance = 0
        self.distance_to_move = distance
        self.path_grid_pos = maze.to_index(self.position)
        self.offset = position

    def move(self, direction: int):
        if direction == 0:
            self.position.y += self.speed
        elif direction == 1:
            self.position.y -= self.speed
        elif direction == 2:
            self.position.x += self.speed
        elif direction == 3:
            self.position.x -= self.speed

    def make_moves(self, cell_size: int):
        if len(self.moves) == 0:
            return

        direction = self.moves[-1]
        self.move(direction)
        self.moved_distance += self.speed
        
        if self.moved_distance < self.distance_to_move:
            return

        self.path_grid_pos = move_grid_index_by_direction(self.path_grid_pos, direction)
        self.position = vec2_from_int_tuple(self.path_grid_pos) * Vec2(cell_size, cell_size)
        self.moves.pop()
        self.moved_distance = 0

    def add_direction(self, new_direction: int):
        self.moves = push_to_direction_list(self.moves, new_direction)

    def render(self, screen: pygame.Surface, offset: Vec2):
        screen.blit(self.image, (self.position + offset).to_tuple())
        
    def get_bounding_box(self) -> Tuple[Vec2, Vec2]:
        return get_bounding_box(self.position, vec2_from_int_tuple(self.size))

def is_collide(bounding_box1: Tuple[Vec2, Vec2], bounding_box2: Tuple[Vec2, Vec2]):
    will_x_collide = bounding_box1[0].x < bounding_box2[1].x and bounding_box2[0].x < bounding_box1[1].x
    will_y_collide = bounding_box1[0].y < bounding_box2[1].y and bounding_box2[0].y < bounding_box1[1].y
    return will_x_collide and will_y_collide

class Button:
    def __init__(self, img: pygame.Surface, text: pygame.Surface, size: Vec2, dest: Vec2, pos: Vec2):
        self.img = pygame.transform.scale(img, size.to_tuple())
        self.img.blit(text, (size / Vec2(2, 2) + dest).to_tuple())
        self.pos = pos

    def render(self, dest: pygame.Surface):
        dest.blit(self.img, self.pos.to_tuple())

def level(cell_size: int, enemy_speed: float, player_speed: float, maze_size: Tuple[int, int]):
    from pygame import font

    window = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    window.fill((0, 0, 0))
    running = True
    keys = Keys({})
    maze = MazeSprite(
        Vec2(cell_size, cell_size),
        {
            "#": pygame.image.load(f"{dir}/imgs/wall.png"),
            "X": pygame.image.load(f"{dir}/imgs/win.png"),
        },
        Maze(maze_size),
        Vec2(0, 0),
    )
    player = Player(maze, speed=player_speed, size=Vec2(cell_size / 4, cell_size / 4), position=Vec2(cell_size, cell_size))
    enemy_position = (Vec2(maze_size[0] - 1, maze_size[1] - 1) * Vec2(2, 2) + Vec2(1, 1)) * Vec2(cell_size, cell_size)
    init_moves = maze.maze.pathfind((maze_size[0] - 1, maze_size[1] - 1), (0, 0))
    init_moves = [move for move in init_moves for _ in range(2)]
    enemy = Enemy(cell_size, maze, speed=enemy_speed, size=Vec2(cell_size, cell_size), position=enemy_position, moves=init_moves)

    font.init()
    font = font.Font(None, 70)
    lose = font.render("LOSE", True, (241, 40, 12))
    win = font.render("WIN", True, (19, 232, 51))
    game_state = GameState.PLAY
    clock = pygame.time.Clock()
    startup = pygame.time.get_ticks()

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                print("quit game")
                running = False
            keys.update(event)

        now = pygame.time.get_ticks()
        offset = -player.position + Vec2(SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2)
        direction = player.on_key(keys, game_state, maze)
        if direction is not None:
            enemy.add_direction(direction)
        if now - startup > 15_000:
            enemy.make_moves(cell_size)
        player.render(window, game_state)
        enemy.render(window, offset)
        maze.render(window, offset)

        game_state = player.collision_detection(maze, game_state)

        player_bb = player.get_bounding_box()
        enemy_bb = enemy.get_bounding_box()
        if is_collide(player_bb, enemy_bb):
            game_state = GameState.LOSE

        if game_state == GameState.LOSE:
            window.blit(lose, lose.get_rect(center=(SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2)))
        if game_state == GameState.WIN:
            window.blit(win, win.get_rect(center=(SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2)))

        pygame.display.flip()
        clock.tick()
        window.fill((0, 0, 0))

pygame.init()
level(32, 0.4, 0.5, (8, 8))
