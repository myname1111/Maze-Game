import random
from copy import deepcopy
from typing import Optional, Tuple

import pygame

from constants import BASE_SCREEN_HEIGHT, BASE_SCREEN_WIDTH, DIR
from entities import Enemy, KillType, Player, is_collide
from game_state import GameState
from input import Keys
from maze import Maze, MazeSprite
from ui import Button, on_mouse_click
from vector import Vec2, vec2_from_int_tuple


def run_level(
    cell_size: int,
    enemy_speed: float,
    player_speed: float,
    maze_size: Tuple[int, int],
    level: int,
) -> Optional[GameState]:
    """
    Run a level with a set of specifications

    Args:
        cell_size (int): How big or small each cell of the maze should be, determines scale as a whole
        enemy_speed (int): How fast the enemy runs in kilopixels per second
        player_speed (int): How fast the player runs in kilopixels per second
        maze_size (int, int): The size of the maze
        level (int): What the current level is, used for displaying text

    Returns:
        Optional[GameState]: It either returns a GameState which describes how the level ended (won or lost), or None which represents quitting the game
    """
    from pygame import font

    window = pygame.display.set_mode(
        (BASE_SCREEN_WIDTH, BASE_SCREEN_HEIGHT), pygame.RESIZABLE
    )
    window.fill((0, 0, 0))
    running = True
    keys = Keys({})
    maze = MazeSprite(
        Vec2(cell_size, cell_size),
        {
            "#": pygame.image.load(f"{DIR}/imgs/wall.png"),
            "X": pygame.image.load(f"{DIR}/imgs/win.png"),
        },
        Maze(maze_size),
        Vec2(0, 0),
    )
    player_path_pos = (0, 1) if maze.maze.depth[1][0] == 1 else (1, 0)
    player_pos = (
        vec2_from_int_tuple(player_path_pos) * Vec2(2, 2) + Vec2(11 / 8, 11 / 8)
    ) * Vec2(cell_size, cell_size)
    player = Player(
        maze,
        speed=player_speed,
        size=Vec2(cell_size / 4, cell_size / 4),
        position=player_pos,
    )
    init_moves = maze.maze.pathfind((0, 0), player_path_pos)
    init_moves = [move for move in init_moves for _ in range(2)]
    enemy = Enemy(
        cell_size,
        maze,
        speed=enemy_speed,
        size=Vec2(cell_size, cell_size),
        position=Vec2(cell_size, cell_size),
        moves=deepcopy(init_moves),
    )

    font.init()
    font_big = font.Font(None, 70)
    lose = font_big.render("YOU LOST", True, (241, 40, 12))
    win = font_big.render("YOU WON", True, (19, 232, 51))
    level_text = font_big.render(f"Level {level}", True, (255, 255, 255))

    font_medium = font.Font(None, 20)
    button_size = Vec2(128, 64)
    button_center = Vec2(
        BASE_SCREEN_WIDTH / 2, BASE_SCREEN_HEIGHT / 2
    ) - button_size / Vec2(2, 2)
    continue_to_next_level = Button(
        pygame.image.load(f"{DIR}/imgs/button.png"),
        font_medium.render("Continue", True, (0, 0, 0)),
        button_size,
        button_center,
    )
    restart_game = Button(
        pygame.image.load(f"{DIR}/imgs/button fail.png"),
        font_medium.render("Restart", True, (0, 0, 0)),
        button_size,
        button_center,
    )

    heart_image = pygame.image.load(f"{DIR}/imgs/heart.png")
    heart_size = (cell_size * 1.5, cell_size * 1.5)
    heart_image = pygame.transform.scale(heart_image, heart_size)

    game_state = GameState.PLAY
    clock = pygame.time.Clock()
    startup = pygame.time.get_ticks()
    now = pygame.time.get_ticks()

    actual_width = BASE_SCREEN_WIDTH
    actual_height = BASE_SCREEN_HEIGHT

    base_window = pygame.surface.Surface((actual_width, actual_height))

    lives = player.lives

    if 0 < level < 5:
        sound = pygame.mixer.Sound(f"{DIR}/sounds/1-5.mp3")
    elif 5 < level < 10:
        sound = pygame.mixer.Sound(f"{DIR}/sounds/5-10.mp3")
    else:
        sound_picker = random.randint(0, 1)
        if sound_picker == 0:
            sound = pygame.mixer.Sound(f"{DIR}/sounds/10A+.mp3")
        if sound_picker == 1:
            sound = pygame.mixer.Sound(f"{DIR}/sounds/10B+.mp3")

    sound.play(-1)

    while running:
        mouse = pygame.mouse.get_pos()
        for event in pygame.event.get():
            keys.update(event)
            if event.type == pygame.QUIT:
                running = False
                return None
            if event.type == pygame.MOUSEBUTTONDOWN:
                new_state = on_mouse_click(
                    game_state, mouse, continue_to_next_level, restart_game
                )
                if new_state == GameState.PLAY:
                    continue
                running = False
            if event.type == pygame.VIDEORESIZE:
                actual_width = event.w
                actual_height = event.h

        # print(enemy.grid_pos)
        # print(enemy.position)
        if game_state == GameState.PLAY:
            prev = now
            now = pygame.time.get_ticks()
            delta_time = now - prev
            offset = -player.position + Vec2(
                BASE_SCREEN_WIDTH / 2, BASE_SCREEN_HEIGHT / 2
            )
            direction = player.on_key(keys, game_state, maze, delta_time)
            if direction is not None:
                enemy.add_direction(direction)
                print(enemy.moves, direction)
            if now - startup > 1_000:
                enemy.make_moves(cell_size, delta_time)
            player.update(now - startup)
            player.render(base_window, game_state)
            enemy.render(base_window, offset)
            maze.render(base_window, offset)

            game_state = player.collision_detection(maze, game_state)

            player_bb = player.get_bounding_box()
            enemy_bb = enemy.get_bounding_box()
            for i in range(player.lives):
                base_window.blit(
                    heart_image, (BASE_SCREEN_WIDTH - heart_size[0] * (i + 1), 0)
                )

            if is_collide(player_bb, enemy_bb):
                game_state = player.kill(KillType.BY_ENEMY)
            if player.lives != lives:
                # new_pos = maze.to_path_index(player.position)
                new_pos = player_path_pos
                player.path_grid_pos = new_pos
                player.position = maze.from_path_index(new_pos) + Vec2(
                    3 * cell_size / 8, 3 * cell_size / 8
                )

                # new_enemy_pos = maze.to_path_index(enemy.position)
                new_enemy_pos = (0, 0)
                # new_enemy_pos = ((enemy.grid_pos[0] // 2 - 1), (enemy.grid_pos[1] - 1) // 2)
                enemy.position = maze.from_path_index(new_enemy_pos)
                enemy.grid_pos = (new_enemy_pos[0] * 2 + 1, new_enemy_pos[1] * 2 + 1)
                enemy.moved_distance = 0

                new_moves = maze.maze.pathfind(new_enemy_pos, new_pos)
                print(new_moves, new_enemy_pos, new_pos)
                # # print(enemy.moves)
                # new_moves += enemy.moves[-2:]
                enemy.moves = deepcopy([move for move in new_moves for _ in range(2)])
                print(enemy.moves, "moves")
                # enemy.moves = deepcopy(init_moves)
                # enemy.moves = new_moves
                # print(init_moves, "what")
                lives = player.lives

                # TODO: Move the enemy to the nearest path cell, then do everything else
        # print(player.position)

        alert_center = (BASE_SCREEN_WIDTH / 2, BASE_SCREEN_HEIGHT / 2 - 75)
        if game_state == GameState.LOSE:
            base_window.blit(lose, lose.get_rect(center=alert_center))
            restart_game.render(base_window)
        if game_state == GameState.WIN:
            base_window.blit(win, win.get_rect(center=alert_center))
            continue_to_next_level.render(base_window)

        base_window.blit(level_text, (0, 0))

        transformed_window = pygame.transform.scale(
            base_window, (actual_width, actual_height)
        )
        window.blit(transformed_window, (0, 0))
        pygame.display.flip()
        clock.tick()
        base_window.fill((0, 0, 0))

    sound.stop()

    return game_state
