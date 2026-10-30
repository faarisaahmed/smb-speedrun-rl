"""Drawing helpers: the agent's-eye block grid and the viewer frame."""
import cv2
import numpy as np

from smb.obs import COLS, HOSTILE, ITEM, MARIO, PLATFORM, ROWS, screen_left

CELL = 12
OBJ_COLORS = {MARIO: (40, 40, 230), HOSTILE: (30, 160, 255), PLATFORM: (60, 200, 60), ITEM: (230, 120, 220)}
TILE_COLOR = (150, 150, 150)
BG = (30, 26, 24)


def grid_image(tiles, objs, cell=CELL):
    """BGR image of the grid: grey = terrain, colored = objects."""
    img = np.full((ROWS * cell, COLS * cell, 3), BG, np.uint8)
    for r in range(ROWS):
        for c in range(COLS):
            color = OBJ_COLORS.get(int(objs[r, c])) or (TILE_COLOR if tiles[r, c] else None)
            if color:
                cv2.rectangle(img, (c * cell + 1, r * cell + 1), ((c + 1) * cell - 1, (r + 1) * cell - 1), color, -1)
    return img


def frame(screen_rgb, tiles, objs, lines, scale=3):
    """Game screen scaled up with the grid view and text lines beside it."""
    game = cv2.resize(cv2.cvtColor(screen_rgb, cv2.COLOR_RGB2BGR), None, fx=scale, fy=scale,
                      interpolation=cv2.INTER_NEAREST)
    h = game.shape[0]
    grid = grid_image(tiles, objs, cell=(h // 2) // ROWS)
    width = grid.shape[1] + 20
    panel = np.full((h, width + (game.shape[1] + width) % 2, 3), BG, np.uint8)  # even width for h264
    panel[10:10 + grid.shape[0], 10:10 + grid.shape[1]] = grid
    y = grid.shape[0] + 50
    for text in lines:
        cv2.putText(panel, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (235, 235, 235), 1, cv2.LINE_AA)
        y += 32
    return np.hstack([game, panel])
