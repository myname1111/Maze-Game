import pygame

from constants import BASE_SCREEN_HEIGHT, BASE_SCREEN_WIDTH
from staging import GameState, run_level

# 0: DOWN
# 1: UP
# 2: RIGHT
# 3: LEFT

pygame.init()
level = 1
window = pygame.display.set_mode(
    (BASE_SCREEN_WIDTH, BASE_SCREEN_HEIGHT), pygame.RESIZABLE
)

while True:
    maze_size = level + 4
    enemy_speed = 0.2 * level / (level + 10)
    delay = 100 / (level + 20) + 5
    out = run_level(32, enemy_speed, 0.1, (maze_size, maze_size), level, delay, window)
    if out is None:
        print("Quitting game")
        break
    if out == GameState.WIN:
        level += 1
    if out == GameState.LOSE:
        level = 1
