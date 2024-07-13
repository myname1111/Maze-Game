from copy import deepcopy
from dataclasses import dataclass
from enum import IntEnum
from typing import Optional, Tuple

import pygame

from constants import BASE_SCREEN_HEIGHT, BASE_SCREEN_WIDTH, DIR
from direction import (get_direction_from_offset, get_opposite_direction,
                       move_grid_index_by_direction)
from game_state import GameState
from input import Keys
from maze import MazeSprite, get_bounding_box
from vector import Vec2, vec2_from_int_tuple


def move_back[T](in_list: Tuple[T, T], new_value: T) -> Tuple[T, T]:
    return (deepcopy(in_list[1]), deepcopy(new_value))


@dataclass
class PrevPos:
    prev_pos: Tuple[Vec2, Vec2]
    pos_updates: int
    prev_pos_update_time: int

    def update(self, time_elapsed: int, new_position: Vec2):
        new_pos_updates = time_elapsed // self.prev_pos_update_time
        if self.pos_updates < new_pos_updates:
            self.pos_updates = new_pos_updates
            self.prev_pos = move_back(self.prev_pos, new_position)


class KillType(IntEnum):
    BY_LAVA = 0
    BY_ENEMY = 1


class Player:

    def __init__(
        self,
        maze: MazeSprite,
        position: Optional[Vec2] = None,
        size: Optional[Vec2] = None,
        speed: float = 1,
        lives: int = 3,
        prev_pos_update_time_lava: int = 1_000,
        prev_pos_update_time_enemy: int = 5_000,
    ) -> None:
        self.position = Vec2(0, 0) if position is None else position
        self.image = pygame.image.load(f"{DIR}/imgs/player.png")
        if size is not None:
            self.image = pygame.transform.scale(self.image, size.to_tuple())
        self.size = self.image.get_size()
        self.speed = speed
        self.path_grid_pos = maze.to_path_index(self.position)
        self.lives = lives
        self.prev_pos_lava = PrevPos(
            (deepcopy(self.position), deepcopy(self.position)),
            0,
            prev_pos_update_time_lava,
        )
        self.prev_pos_enemy = PrevPos(
            (deepcopy(self.position), deepcopy(self.position)),
            0,
            prev_pos_update_time_enemy,
        )

    def on_key(
        self, key: Keys, state: GameState, maze: MazeSprite, delta_time: int
    ) -> Optional[int]:
        if state == GameState.LOSE:
            return

        delta_pos = self.speed * delta_time

        if key.pressed(pygame.K_LEFT):
            self.position.x -= delta_pos
        if key.pressed(pygame.K_RIGHT):
            self.position.x += delta_pos
        if key.pressed(pygame.K_DOWN):
            self.position.y += delta_pos
        if key.pressed(pygame.K_UP):
            self.position.y -= delta_pos

        new_path_grid_pos = maze.to_path_index(self.position * Vec2(2, 2))
        if new_path_grid_pos == self.path_grid_pos:
            return None

        offset = (
            new_path_grid_pos[0] - self.path_grid_pos[0],
            new_path_grid_pos[1] - self.path_grid_pos[1],
        )
        direction = get_direction_from_offset(offset)

        self.path_grid_pos = new_path_grid_pos

        return direction

    def collide_with_cell(
        self, cell: str, init_state: GameState
    ) -> Tuple[GameState, bool]:
        is_killed = False
        new_state = init_state

        match cell:
            case " ":
                pass
            case "#":
                is_killed = True
                if self.lives <= 1:
                    new_state = GameState.LOSE
                new_state = GameState.PLAY
            case "X":
                new_state = GameState.WIN
            case _:
                assert False

        return (new_state, is_killed)

    def collision_detection(self, maze: MazeSprite, init_state: GameState) -> GameState:
        collided_cells = maze.collide_with_sprite(
            self.position, vec2_from_int_tuple(self.image.get_size())
        )
        state = init_state
        is_killed = False
        for collided_cell in collided_cells:
            (new_state, is_killed_in_cell) = self.collide_with_cell(
                collided_cell, state
            )
            state = state.combine_state(new_state)
            is_killed |= is_killed_in_cell
        if is_killed:
            state = state.combine_state(self.kill(KillType.BY_LAVA))
        return state

    def update(self, time_elapsed: int):
        self.prev_pos_lava.update(time_elapsed, self.position)
        self.prev_pos_enemy.update(time_elapsed, self.position)

    def kill(self, kill_type: KillType) -> GameState:
        if kill_type == KillType.BY_LAVA:
            self.position = deepcopy(self.prev_pos_lava.prev_pos[0])
        if kill_type == KillType.BY_ENEMY:
            self.position = deepcopy(self.prev_pos_enemy.prev_pos[0])
        self.prev_pos_lava.prev_pos = (deepcopy(self.position), deepcopy(self.position))
        self.lives -= 1

        if self.lives <= 0:
            return GameState.LOSE
        else:
            return GameState.PLAY

    def render(self, screen: pygame.Surface, state: GameState):
        if state == GameState.LOSE:
            return

        screen.blit(self.image, (BASE_SCREEN_WIDTH / 2, BASE_SCREEN_HEIGHT / 2))

    def get_bounding_box(self) -> Tuple[Vec2, Vec2]:
        return get_bounding_box(self.position, vec2_from_int_tuple(self.size))


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
        moves: Optional[list[int]] = None,
    ):
        self.position = Vec2(0, 0) if position is None else position
        self.image = pygame.image.load(f"{DIR}/imgs/enemy.png")
        if size is not None:
            self.image = pygame.transform.scale(self.image, size.to_tuple())
        self.size = self.image.get_size()
        self.speed = speed
        self.moves = [] if moves is None else moves
        self.moved_distance = 0
        self.distance_to_move = distance
        self.grid_pos = maze.to_index(self.position)

    def move(self, direction: int, delta_time: int):
        delta_pos = self.speed * delta_time
        if direction == 0:
            self.position.y += delta_pos
        elif direction == 1:
            self.position.y -= delta_pos
        elif direction == 2:
            self.position.x += delta_pos
        elif direction == 3:
            self.position.x -= delta_pos

    def make_moves(self, cell_size: int, delta_time: int):
        if len(self.moves) == 0:
            return

        direction = self.moves[-1]
        self.move(direction, delta_time)
        self.moved_distance += delta_time * self.speed

        if self.moved_distance < self.distance_to_move:
            return

        self.grid_pos = move_grid_index_by_direction(self.grid_pos, direction)
        self.position = vec2_from_int_tuple(self.grid_pos) * Vec2(cell_size, cell_size)
        self.moves.pop()
        self.moved_distance = 0

    def add_direction(self, new_direction: int):
        self.moves = push_to_direction_list(self.moves, new_direction)

    def render(self, screen: pygame.Surface, offset: Vec2):
        screen.blit(self.image, (self.position + offset).to_tuple())

    def get_bounding_box(self) -> Tuple[Vec2, Vec2]:
        return get_bounding_box(self.position, vec2_from_int_tuple(self.size))

    def path_pos_on_finish(self) -> Tuple[int, int]:
        out = self.grid_pos
        for direction in self.moves:
            out = move_grid_index_by_direction(out, direction)
        return out


def is_collide(bounding_box1: Tuple[Vec2, Vec2], bounding_box2: Tuple[Vec2, Vec2]):
    will_x_collide = (
        bounding_box1[0].x < bounding_box2[1].x
        and bounding_box2[0].x < bounding_box1[1].x
    )
    will_y_collide = (
        bounding_box1[0].y < bounding_box2[1].y
        and bounding_box2[0].y < bounding_box1[1].y
    )
    return will_x_collide and will_y_collide
