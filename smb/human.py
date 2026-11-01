"""The human world-record run as a Luigi ghost racing the agent in the videos.

Its path comes from track_human.py (runs/<run>/human/<level>.json): Mario's
world (x, y) per NES frame, traced from a video of the human run.
"""
import json
import os

import cv2
import numpy as np

NES_FPS = 60.0988
# Mario's palette -> Luigi's: green shirt and cap, navy overalls (RGB).
LUIGI_COLORS = {(216, 40, 0): (40, 176, 40), (136, 112, 0): (32, 56, 136)}
MAX_GAP = 4   # frames: draw the nearest tracked position up to this far away


def luigi(sprite_bgr, mask):
    """Recolor a Mario sprite (BGR float, from ghosts.mario_sprite) as Luigi."""
    out = sprite_bgr.copy()
    for mario, l in LUIGI_COLORS.items():
        hit = np.all(np.abs(sprite_bgr - np.array(mario[::-1], np.float32)) < 1, axis=2)
        out[hit] = l[::-1]
    return out, mask


class HumanGhost:
    def __init__(self, run_dir, level):
        path = os.path.join(run_dir, "human", f"{level}.json")
        data = json.load(open(path)) if os.path.exists(path) else None
        self.ok = data is not None
        if not self.ok:
            return
        self.x, self.y = data["x"], data["y"]
        self.flag_frame = data["flag_frame"]
        self.source = data.get("source", "human")

    def at(self, frame):
        """(x, y) in episode units at this NES frame of the level, or None."""
        if not self.ok or frame >= len(self.x) + MAX_GAP:
            return None
        for d in range(MAX_GAP + 1):
            for f in (frame - d, frame + d):
                if 0 <= f < len(self.x) and self.x[f] is not None:
                    return self.x[f], self.y[f]
        return None

    def label(self):
        if not self.ok:
            return None
        t = f"{self.flag_frame / NES_FPS:.2f}s" if self.flag_frame else "--"
        return f"Luigi: {self.source} {t}"

    def overlay(self, screen_rgb, frame, left, sprite, mask):
        """The stable-retro screen (RGB, world x of column 0 = left + 8) with Luigi drawn on it."""
        pos = self.at(frame)
        if pos is None:
            return screen_rgb
        x, y = pos
        img = cv2.cvtColor(np.ascontiguousarray(screen_rgb), cv2.COLOR_RGB2BGR)
        draw(img, sprite, mask, x - left - 8, y - 248, 1.0)   # his feet are 24 rows below Player_Y on screen
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def draw(view, sprite, mask, sx, sy, alpha):
    h, w = mask.shape
    x0, y0 = max(sx, 0), max(sy, 0)
    x1, y1 = min(sx + w, view.shape[1]), min(sy + h, view.shape[0])
    if x0 >= x1 or y0 >= y1:
        return
    m = mask[y0 - sy:y1 - sy, x0 - sx:x1 - sx]
    region = view[y0:y1, x0:x1].astype(np.float32)
    region[m] = region[m] * (1 - alpha) + sprite[y0 - sy:y1 - sy, x0 - sx:x1 - sx][m] * alpha
    view[y0:y1, x0:x1] = region.astype(view.dtype)
