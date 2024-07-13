from typing import Optional, Tuple


def move_grid_index_by_direction(
    index: Tuple[int, int], direction: int
) -> Tuple[int, int]:
    """Move an index on the grid by a direction"""
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


def move_position_by_direction(
    index: Tuple[int, int], direction: int
) -> Tuple[int, int]:
    """Move an position by a direction"""
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


def get_opposite_direction(dir: int) -> int:
    """Get the opposite direction"""
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


def get_direction_from_offset(offset: Tuple[int, int]) -> Optional[int]:
    """Gets a direction given an offset in a position"""
    if offset == (0, 1):
        return 0
    elif offset == (0, -1):
        return 1
    elif offset == (1, 0):
        return 2
    elif offset == (-1, 0):
        return 3
