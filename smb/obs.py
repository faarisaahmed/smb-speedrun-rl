"""What the agent sees: the visible screen as a block grid read from RAM.

  tiles    (ROWS, COLS)     raw metatile id of every visible block
  objs     (ROWS, COLS)     object category per cell (Mario, enemy, platform, item)
  scalars  (N_SCALARS,)     Mario's exact position/speed/state + each object's
                            exact offset from Mario, so sub-block precision
                            (needed for tight jumps and clips) isn't lost

The grid is 16x16-px blocks aligned to the block grid, starting at the block
containing the screen's left edge. Verified against screenshots: block row r
covers NES screen y 32+16r, Mario's top is at Player_Y+16, enemies' at Enemy_Y+8.
"""
import numpy as np

ROWS, COLS = 13, 17

EMPTY, MARIO, HOSTILE, PLATFORM, ITEM = range(5)
N_OBJ = 5

# Enemy slot RAM (5 enemy slots + slot 5 = powerup).
ENEMY_FLAG, ENEMY_ID, ENEMY_PAGE, ENEMY_X, ENEMY_Y = 0x0F, 0x16, 0x6E, 0x87, 0xCF
N_SLOTS = 6
# Hammer Bros' hammers live in the 9 "misc" object slots, not the enemy slots.
MISC_STATE, MISC_PAGE, MISC_X, MISC_Y = 0x2A, 0x7A, 0x93, 0xDB
N_MISC = 9
HAMMER_BRO = 0x05

PLATFORM_IDS = set(range(0x24, 0x2D)) | {0x32}            # lifts, balance lifts, spring
ITEM_IDS = {0x2E, 0x2F, 0x30, 0x31, 0x34, 0x35}            # powerup, vine, flags, warp, toad
SMALL_PLATFORM_IDS = {0x2B, 0x2C}

N_MARIO_SCALARS = 11
N_SCALARS = N_MARIO_SCALARS + N_SLOTS * 4


def signed(b):
    b = int(b)
    return b - 256 if b > 127 else b


def screen_left(ram):
    return int(ram[0x071A]) * 256 + int(ram[0x071C])


def observe(ram):
    ram = ram.astype(np.int32)
    left = screen_left(ram)
    c0 = left // 16

    cols = (c0 + np.arange(COLS)) % 32
    base = 0x500 + (cols // 16) * 0xD0 + cols % 16
    tiles = ram[base[None, :] + 16 * np.arange(ROWS)[:, None]].astype(np.float32)

    objs = np.zeros((ROWS, COLS), np.float32)

    def mark(x, top, rows, width, cat):
        r0 = (top - 32) // 16
        cs = range((x + 8) // 16 - c0, (x + 8 + width - 16) // 16 - c0 + 1)
        for r in range(r0, r0 + rows):
            for c in cs:
                if 0 <= r < ROWS and 0 <= c < COLS:
                    objs[r, c] = cat

    mx = ram[0x6D] * 256 + ram[0x86]
    my = ram[0xB5] * 256 + ram[0xCE] - 256   # screen y; negative = above the screen
    big = ram[0x0754] == 0                    # PlayerSize: 0 big, 1 small
    crouch = ram[0x0714] != 0

    slots = []
    for i in range(N_SLOTS):
        if not ram[ENEMY_FLAG + i]:
            slots.append([0.0, 0.0, 0.0, 0.0])
            continue
        eid = ram[ENEMY_ID + i]
        ex = ram[ENEMY_PAGE + i] * 256 + ram[ENEMY_X + i]
        ey = ram[ENEMY_Y + i]
        if i == 5 or eid in ITEM_IDS:
            cat, width = ITEM, 16
        elif eid in PLATFORM_IDS:
            cat, width = PLATFORM, 32 if eid in SMALL_PLATFORM_IDS else 48
        else:
            cat, width = HOSTILE, 16
        mark(ex, ey + 8, 1, width, cat)
        slots.append([cat / (N_OBJ - 1), (ex - mx) / 256, (ey - my) / 256, 1.0])

    # Misc slots also hold coins popped from blocks, so only read them with a
    # Hammer Bro around. Before this the agent couldn't see hammers at all.
    if any(ram[ENEMY_FLAG + i] and ram[ENEMY_ID + i] == HAMMER_BRO for i in range(5)):
        for k in range(N_MISC):
            if ram[MISC_STATE + k]:
                mark(ram[MISC_PAGE + k] * 256 + ram[MISC_X + k], ram[MISC_Y + k], 1, 16, HOSTILE)

    if big and not crouch:
        mark(mx, my, 2, 16, MARIO)
    else:
        mark(mx, my + 16, 1, 16, MARIO)

    mario = [
        (mx - left) / 256,              # where on screen
        my / 256,
        signed(ram[0x57]) / 48,         # x speed
        signed(ram[0x9F]) / 8,          # y speed
        ram[0x1D] / 3,                  # 0 ground, 1 jumping, 2 falling, 3 flagpole
        ram[0x0756] / 2,                # powerup: small/big/fire
        float(big),
        float(crouch),
        (mx % 16) / 16,                 # sub-block offsets
        (my % 16) / 16,
        (left % 16) / 16,               # how the grid is shifted against the screen
    ]
    scalars = np.array(mario + [v for s in slots for v in s], np.float32)
    return tiles, objs, scalars
