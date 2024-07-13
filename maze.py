import random
from typing import Optional, Tuple

import pygame

from direction import (get_opposite_direction, move_grid_index_by_direction,
                       move_position_by_direction)
from vector import Vec2


def get_cell_in_grid(paths: list[list[Optional[int]]], index: Tuple[int, int]):
    """
    Gets a cell from a grid

    Args:
        paths ([[Optional[int]]]): The grid to get it from
        index (int, int): The index of the location

    Returns:
        Optional[int]: The cell in question
    """
    return paths[index[1]][index[0]]


def unwrap[T](input_val: Optional[T]) -> T:
    """
    Get the inner value of an optional, throwing an error if it is None

    Args:
        input_val (Optional[T]): The input value, please be confident that this value isn't be None

    Returns:
        T: The output value
    """
    if input_val is not None:
        return input_val
    else:
        assert False


def is_valid_cell_to_go_to(
    grid: list[list[Optional[int]]], cell_in_direction: Tuple[int, int]
) -> bool:
    """
    Checks whether or not a cell in a grid can be travelled to in the traversal

    Args:
        grid ([[Optional[int]]]): The grid
        cell_in_direction (int, int): The index of the cell to be checked

    Returns:
        bool: Represents if the traversal can go to the cell
    """
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
    """
    Go to the next cell for a randomised DFS traversal algorithm

    It first checks all the neighbouring cells. If there exist one that is not
    connected to any other cell, then the next cell will be that cell and it
    will be connected to the previous cell. If there isn't, the next cell will
    be the previous cell that is connected with the current cell.

    Args:
        curr_cell (int, int): The index of the current cell
        grid: [[Optional[int]]]: The grid containing the cell and the next cell

    Returns:
        (int, int): The index of the next cell
        Optional[int]: The direction of the next cell, returns none if the
            algorithm backtracks
    """
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


def init_grid(x: int, y: int) -> list[list[Optional[int]]]:
    """Initialise a grid of paths"""

    out = []
    for _ in range(0, y):
        row: list[Optional[int]] = []
        for _ in range(0, x):
            row.append(None)
        out.append(row)
    return out


def create_paths(size: Tuple[int, int]) -> list[list[Optional[int]]]:
    """
    Create the paths of a maze with a root at (0, 0)

    Args:
        size (int, int): The size of the maze

    Returns:
        [[Optional[int]]]: The resulting maze, the root has no direction and is
            thus None
    """
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
    """
    Initialise a maze represented with characters

    Args:
        size (int, int): The size of the maze

    Returns:
        [[str]]: An empty maze
    """
    maze = []
    for x in range(size[0] * 2 + 1):
        row = []
        for y in range(size[1] * 2 + 1):
            row.append(" " if x % 2 == 1 and y % 2 == 1 else "#")
        maze.append(row)
    return maze


def create_walls_from_paths(
    size: Tuple[int, int], paths: list[list[Optional[int]]], win: str = "X"
) -> str:
    """
    Turns a grid of paths created from the randomised DFS traversal algorithm
    into a maze with walls

    Args:
        size (int, int): The size of the maze to be created
        paths [[Optional[int]]]: The grid of paths which the maze will be
            derived from
        win (str): The character to represent the win cell

    Returns:
        str: The maze represented as a single continous string
    """
    maze = init_maze(size)
    for y, row in enumerate(paths):
        for x, cell in enumerate(row):
            if cell is None:
                continue

            (x_new, y_new) = move_grid_index_by_direction((x * 2 + 1, y * 2 + 1), cell)

            maze[y_new][x_new] = " "

    maze[size[1] * 2 - 1][size[0] * 2] = win
    return "".join(["".join(row) for row in maze])


def step_back(
    pos: Tuple[int, int], paths: list[list[Optional[int]]]
) -> Tuple[Tuple[int, int], int] | None:
    """
    Steps back towards the root in a maze path with a root

    Args:
        pos (int, int): The position of the cell to step back on
        paths ([[Optional[int]]]): A grid of connections where the cell is from

    Returns:
        (
            (int, int): The index of the previous cell
            int: The direction from the current cell to the previous cell
        ) |
        None: No change as it has hit the root
    """
    direction = paths[pos[1]][pos[0]]
    if direction is None:
        return None

    new = move_grid_index_by_direction(pos, direction)
    return (new, direction)


def many_step_back(
    pos: Tuple[int, int], paths: list[list[Optional[int]]], count: int
) -> Tuple[Tuple[int, int], list[int]]:
    """
    Take many steps back towards the root in a maze path with a root

    Args:
        pos (int, int): The position of the cell to step back on
        paths ([[Optional[int]]]): A grid of connections where the cell is from
        count (int): How many steps back should it take

    Returns:
        (int, int): The index of the output cell
        [int]: The directions from the current cell to the output cell
    """
    curr_pos = pos
    out = []
    for _ in range(count):
        new = step_back(curr_pos, paths)
        if new is None:
            return (curr_pos, out)
        curr_pos = new[0]
        out.append(new[1])
    return (curr_pos, out)


def get_depth(
    paths: list[list[Optional[int]]], size: Tuple[int, int]
) -> list[list[int]]:
    """
    Gets the depth or how far away each cell is from the root in a grid of
    connections

    Args:
        paths ([[Optional[int]]]): The grid of connections
        size (int, int): The size of that grid

    Returns:
        [[int]]: A grid representing the depth of every cell
    """
    out = []
    for y, row in enumerate(paths):
        out_row = []
        for x, _ in enumerate(row):
            out_row.append(len(many_step_back((x, y), paths, size[0] * size[1])[1]))
        out.append(out_row)
    return out


class Maze:
    """
    A class representing the inner irepresntation of a maze

    Attributes:
        cols (int): The collumns of a maze
        rows (int): The rows of a maze
        paths ([[Optional[int]]]): A grid of connections, with None
            representing the root node
        depth ([[int]]): A grid of integers representing the depth, or how far away
            a node is from the root node
        maze (str): The actual maze, using a string to represent different cell
            types
    """

    def __init__(self, size: Tuple[int, int]):
        """
        Initialises a maze

        Args:
            size (int, int): Size of the maze

        Returns:
            Maze: The maze
        """

        self.cols = size[0] * 2 + 1
        self.rows = size[1] * 2 + 1
        self.paths = create_paths(size)
        self.depth = get_depth(self.paths, size)
        self.maze = create_walls_from_paths(size, self.paths)

    def pathfind(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> list[int]:
        """
        Pathfinds from one position to the next

        Args:
            pos1 (int, int): The starting position
            pos2 (int, int): The ending position

        Returns:
            [int]: A list of directions
        """
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
        """Gets a cell in the maze given a row and collumn, cells outside will use a space"""

        if row >= self.rows:
            return " "
        if col >= self.cols:
            return " "

        return self.maze[col + self.cols * row]


class MazeSprite:
    """
    A maze as an object in the game

    Attributes:
        cell_size (Vec2): The size of each cell in pixels
        cell_image ({str: pygame.Surface}): A dictionary mapping each character
            to its texture
        maze (Maze): The abstract representation of the maze
        offset (Vec2): How much shoulld the maze be offsetted
    """

    def __init__(
        self,
        cell_size: Vec2,
        cell_image: dict[str, pygame.Surface],
        maze: Maze,
        offset: Vec2,
    ) -> None:
        """
        Initialises the maze

        Args:
            cell_size (Vec2): The size of each cell in pixels
            cell_image ({str: pygame.Surface}): A dictionary mapping each character
                to its texture
            maze (Maze): The abstract representation of the maze
            offset (Vec2): How much shoulld the maze be offsetted

        Returns:
            MazeSprite: The maze
        """
        self.cell_size = cell_size
        self.cell_image = {
            cell: pygame.transform.scale(cell_image[cell], self.cell_size.to_tuple())
            for cell in cell_image
        }
        self.maze = maze
        self.offset = offset

    def to_index(self, pos: Vec2) -> Tuple[int, int]:
        """Turns a position in pixels to an index to get cells in the maze"""
        out = (pos - self.offset) // self.cell_size
        return (int(out.x), int(out.y))

    def from_index(self, pos: Tuple[int, int]) -> Vec2:
        """Turns an index into a position in pixels"""
        return Vec2(pos[0], pos[1]) * self.cell_size + self.offset

    def to_path_index(self, pos: Vec2) -> Tuple[int, int]:
        """Turns a position in pixels to an index for use in pathfinding"""
        sprite_grid_pos = self.to_index(pos)
        return ((sprite_grid_pos[0]) // 2, sprite_grid_pos[1] // 2)

    def from_path_index(self, pos: Tuple[int, int]) -> Vec2:
        """Turns an index for use in pathfinding into a position in pixels"""
        return Vec2(pos[0] * 2 + 1, pos[1] * 2 + 1) * self.cell_size + self.offset

    def collide_with_sprite(self, other_pos: Vec2, other_size: Vec2) -> list[str]:
        """
        Checks whether or not another sprite will collide with the maze

        Args:
            other_pos (Vec2): The other's sprite's position
            other_size (Vec2): The other's sprite's size

        Returns:
            [str]: A list of cells the maze collides with, represented as strings
        """
        points = get_bounding_box(other_pos, other_size)
        grid_index_bounding_box = [self.to_index(point) for point in points]
        indicies = get_list_of_indicies_inside_grid_index_bounding_box(
            grid_index_bounding_box[0], grid_index_bounding_box[1]
        )
        return [self.maze.get(index[0], index[1]) for index in indicies]

    def render(self, screen: pygame.Surface, offset: Vec2):
        """Renders the maze"""
        for idx, cell in enumerate(self.maze.maze):
            x = idx % self.maze.cols
            y = idx // self.maze.cols

            if cell in self.cell_image:
                screen.blit(
                    self.cell_image[cell], (self.from_index((x, y)) + offset).to_tuple()
                )


def get_list_of_indicies_inside_grid_index_bounding_box(
    start: Tuple[int, int], to: Tuple[int, int]
) -> list[Tuple[int, int]]:
    """Given a start and end index, it computes all indicies inside the range"""
    return [
        (x, y) for y in range(start[1], to[1] + 1) for x in range(start[0], to[0] + 1)
    ]


def get_bounding_box(pos: Vec2, size: Vec2) -> Tuple[Vec2, Vec2]:
    """Turns a position and size into an AABB bounding box"""
    return (pos, pos + size)
