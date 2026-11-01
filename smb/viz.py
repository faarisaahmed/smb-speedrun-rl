"""Drawing helpers: the agent's-eye block grid and the viewer frame."""
import cv2
import numpy as np

from smb.obs import COLS, HOSTILE, ITEM, MARIO, PLATFORM, ROWS, screen_left

CELL = 12
OBJ_COLORS = {MARIO: (40, 40, 230), HOSTILE: (30, 160, 255), PLATFORM: (60, 200, 60), ITEM: (230, 120, 220)}
TILE_COLOR = (150, 150, 150)
BG = (30, 26, 24)
PAD_OFF, PAD_ON = (70, 70, 70), (245, 245, 245)
FACE_OFF, FACE_ON = (40, 40, 95), (50, 50, 235)


def grid_image(tiles, objs, cell=CELL):
    """BGR image of the grid: grey = terrain, colored = objects."""
    img = np.full((ROWS * cell, COLS * cell, 3), BG, np.uint8)
    for r in range(ROWS):
        for c in range(COLS):
            color = OBJ_COLORS.get(int(objs[r, c])) or (TILE_COLOR if tiles[r, c] else None)
            if color:
                cv2.rectangle(img, (c * cell + 1, r * cell + 1), ((c + 1) * cell - 1, (r + 1) * cell - 1), color, -1)
    return img


def draw_pad(img, x, y, buttons, unit=1.0):
    """Draw a NES controller at (x, y) (top-left, BGR) with the held buttons lit."""
    u = lambda v: int(round(v * unit))
    cx, cy = x + u(36), y + u(36)
    arm = u(12)
    cv2.rectangle(img, (x, y), (x + u(190), y + u(72)), (12, 12, 12), -1)
    for name, (dx, dy) in {"LEFT": (-2, 0), "RIGHT": (2, 0), "UP": (0, -2), "DOWN": (0, 2)}.items():
        px, py = cx + dx * arm, cy + dy * arm
        cv2.rectangle(img, (px - arm, py - arm), (px + arm, py + arm),
                      PAD_ON if name in buttons else PAD_OFF, -1)
    cv2.rectangle(img, (cx - arm, cy - arm), (cx + arm, cy + arm), PAD_OFF, -1)
    for name, bx in (("B", u(115)), ("A", u(160))):
        cv2.circle(img, (x + bx, cy), u(17), FACE_ON if name in buttons else FACE_OFF, -1)
        cv2.putText(img, name, (x + bx - u(7), cy + u(7)), cv2.FONT_HERSHEY_SIMPLEX, 0.6 * unit,
                    (255, 255, 255) if name in buttons else (150, 150, 150), max(1, u(2)), cv2.LINE_AA)


def frame(screen_rgb, tiles, objs, lines, scale=3, buttons=None):
    """Game screen scaled up with the grid view, text lines and held buttons beside it."""
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
    if buttons is not None:
        draw_pad(panel, 12, h - 112, buttons, unit=1.3)
    return np.hstack([game, panel])
