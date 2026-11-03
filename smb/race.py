"""A published TAS as a ghost racing the agent in the videos.

Its path comes from track_tas.py (runs/<run>/tas/<level>.json): Mario's
(x, y) per NES frame from the level's start, read from RAM while the movie
plays in our emulator.
"""
import json
import os

import cv2
import numpy as np

NES_FPS = 60.0988
TAS_COLOR = (255, 255, 255)   # BGR: a white silhouette, unlike the agent's own Marios
TAS_ALPHA = 0.7


def tas_look(sprite_bgr, mask):
    """A Mario sprite (BGR float, from ghosts.mario_sprite) as the TAS ghost."""
    out = sprite_bgr.copy()
    out[mask] = TAS_COLOR
    return out, mask


def tas_sprites(env, mario_sprite):
    """{big: (sprite, mask)}: the TAS is usually big or fire Mario, the agent small."""
    return {big: tas_look(*mario_sprite(env, big)) for big in (False, True)}


class TasGhost:
    def __init__(self, run_dir, level):
        path = os.path.join(run_dir, "tas", f"{level}.json")
        data = json.load(open(path)) if os.path.exists(path) else None
        self.ok = data is not None
        if not self.ok:
            return
        self.x, self.y = data["x"], data["y"]
        self.big = data.get("big") or [False] * len(self.x)
        self.flag_frame = data["flag_frame"]
        self.source = data.get("source", "TAS")

    def at(self, frame):
        """(x, y, big) in episode units at this NES frame of the level, or None (another area, or done)."""
        if not self.ok or not 0 <= frame < len(self.x) or self.x[frame] is None:
            return None
        return self.x[frame], self.y[frame], self.big[frame]

    def label(self):
        if not self.ok:
            return None
        t = f"{self.flag_frame / NES_FPS:.2f}s" if self.flag_frame is not None else "--"
        return f"TAS: {self.source} {t}"

    def overlay(self, screen_rgb, frame, left, sprites):
        """The stable-retro screen (RGB, world x of column 0 = left + 8) with the ghost drawn on it."""
        pos = self.at(frame)
        if pos is None:
            return screen_rgb
        x, y, big = pos
        sprite, mask = sprites[big]
        img = cv2.cvtColor(np.ascontiguousarray(screen_rgb), cv2.COLOR_RGB2BGR)
        # His feet are 24 rows below Player_Y on screen, whatever his size.
        draw(img, sprite, mask, x - left - 8, y - 232 - mask.shape[0], TAS_ALPHA)
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
