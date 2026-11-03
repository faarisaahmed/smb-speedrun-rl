"""NES RAM addresses used by Super Mario Bros."""


class RAM:
    PLAYER_STATE = 0x000E  # 0x06 dead, 0x08 normal, 0x0B dying
    FLOAT_STATE = 0x001D   # 0x03 = sliding down the flagpole
    X_PAGE = 0x006D
    X_SCREEN = 0x0086
    Y_PAGE = 0x00B5        # >1 means Mario fell below the screen
    Y_SCREEN = 0x00CE
    X_SPEED = 0x0057
    POWERUP = 0x0756       # 0 small, 1 big, 2 fire
    PLAYER_SIZE = 0x0754   # 0 big, 1 small
    LIVES = 0x075A
    STAGE = 0x075C
    WORLD = 0x075F
    AREA = 0x0760
    AREA_TYPE = 0x074E     # 0 water, 1 ground, 2 underground, 3 castle
    AREA_INDEX = 0x074F    # which area of that type; (type, index) identifies the area
    GAME_MODE = 0x0770     # 0 title/demo, 1 playing, 2 end of world, 3 game over
    TIMER = 0x07F8         # three BCD-ish digits, 0x07F8..0x07FA

    DYING = 0x0B
    DEAD = 0x06
    FLAGPOLE = 0x03


def x_pos(ram):
    return int(ram[RAM.X_PAGE]) * 256 + int(ram[RAM.X_SCREEN])


def y_pos(ram):
    # Larger = higher on screen, measured from the bottom of the world.
    return int(ram[RAM.Y_PAGE]) * 256 + int(ram[RAM.Y_SCREEN])


def game_timer(ram):
    return int(ram[RAM.TIMER]) * 100 + int(ram[RAM.TIMER + 1]) * 10 + int(ram[RAM.TIMER + 2])
