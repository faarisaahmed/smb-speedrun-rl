"""Trace a human run from a video into per-level ghost paths.

Humans' button presses aren't published, only videos, so this follows Mario
through the footage frame by frame:
  - the status bar's WORLD x-y says which level it is (glyphs read from our own emulator),
  - the playfield is matched against that level's panorama to get the scroll,
  - Mario is found by matching his sprites (every pose, small/big/fire, recorded
    from our emulator and shifted into the video's colors) near where he was.

Writes runs/human/<level>.json: Mario's world (x, y) per NES frame from the
level's first frame of play (null where he couldn't be found, e.g. in a bonus
room), in the same units as the training episodes' x/y, plus the frame he
reached the flagpole.

  python track_human.py --video videos/source/warpless_wr.mp4 --preview 3000
      # saves that frame (with a pixel grid) to pick the game area
  python track_human.py --video videos/source/warpless_wr.mp4 --crop X,Y,W,H
      # X,Y,W,H: the full 256x240 NES picture inside the video frame
"""
import argparse
import json
import os

import cv2
import imageio
import numpy as np

from ghosts import MARIO_COLORS, TOP, build_panorama
from smb.env import SMBEnv
from smb.obs import screen_left
from smb.ram import RAM, x_pos

NES_FPS = 60.0988
CROP = 8                  # stable-retro's screen drops 8 px off every side of the 256x240 picture
PLAY_ROW = TOP - CROP     # first screen row below the status bar
LEVEL_CHARS = (16, 144)   # screen (row, col) of the "W-L" under WORLD, 3 glyphs 8 px wide
SEARCH = (-12, 20)        # scroll change per frame we look for (px)
MIN_SCROLL_MATCH = 0.5    # correlation below this: not this level's main area (pipe room, transition)
# Mario's colors in our emulator: small/big (red, skin, brown) and fire (white, red, skin).
MARIO_PALETTE = np.array(MARIO_COLORS + [(252, 252, 252)], np.float32)
START = (40, 176)         # Mario's (x, feet row) on every level's first frame
SPRITE_W, SPRITE_H = 20, 32   # template box: Mario's 16 px plus 2 each side, tall enough for big Mario
REACH = 24                # px from where Mario should be that a match may be, plus...
REACH_PER_LOST = 6        # ...this much per frame he's been lost
MAX_RMS = 45              # color error (0-255) above which a match isn't Mario
LEVELS = ["1-1", "1-3", "2-1", "2-3", "3-1", "3-2", "3-3", "4-1", "4-3", "5-1",
          "5-2", "6-2", "6-3", "7-1", "7-3", "8-1", "8-3"]


def nes_screen(frame, crop):
    """The video frame's game area as a stable-retro-sized RGB screen (224x240)."""
    x, y, w, h = crop
    nes = cv2.resize(frame[y:y + h, x:x + w], (256, 240), interpolation=cv2.INTER_AREA)
    return nes[CROP:240 - CROP, CROP:256 - CROP]


def hud_bits(screen):
    """The three status-bar level glyphs as booleans (white text on any background)."""
    r, c = LEVEL_CHARS
    g = cv2.cvtColor(np.ascontiguousarray(screen[r:r + 8, c:c + 24]), cv2.COLOR_RGB2GRAY).astype(np.float32)
    return g > (g.max() + g.min()) / 2 if g.max() - g.min() > 60 else np.zeros_like(g, bool)


def glyph_library(env):
    """{char: 8x8 bits} for world digits 1-8, level digits 1-3 and '-', from our levels' start screens."""
    glyphs = {}
    for level in LEVELS:
        env.reset(options={"level": level})
        bits = hud_bits(env.screen)
        for i, ch in enumerate(level):
            glyphs.setdefault(ch, bits[:, 8 * i:8 * i + 8])
    return glyphs


def read_level(screen, glyphs):
    """'W-L' shown on the status bar, or None if any glyph is unclear."""
    bits = hud_bits(screen)
    out = ""
    for i in range(3):
        cell = bits[:, 8 * i:8 * i + 8]
        ch, miss = min(((ch, np.sum(cell != g)) for ch, g in glyphs.items()), key=lambda t: t[1])
        if miss > 6:
            return None
        out += ch
    return out


def mario_sprites(env, runs, panoramas):
    """Every Mario pose seen (twice or more) replaying runs as small and big Mario,
    plus fire-colored copies.

    His pixels are those in his colors that differ from the level's panorama
    (bricks and ground share his browns). Returns [(image, mask, dx, dfeet)]:
    each pose trimmed to its pixels, with his x and feet row relative to its corner.
    """
    counts, poses = {}, {}
    for level, actions in runs:
        for powerup in (0, 1):
            env.reset(options={"level": level})
            for a in actions:
                mem = env.env.unwrapped.data.memory
                mem.assign(RAM.POWERUP, "|u1", powerup)
                mem.assign(0x0754, "|u1", 1 - powerup)   # PlayerSize: 0 big, 1 small
                _, _, term, trunc, _ = env.step(int(a))
                if term or trunc:
                    break
                ram = env.emu.get_ram()
                col = x_pos(ram) - screen_left(ram) - CROP - 2
                feet = int(ram[RAM.Y_SCREEN]) + 24   # screen row below his feet
                if ram[RAM.Y_PAGE] != 1 or col < 0 or col + SPRITE_W > 240 or feet - SPRITE_H < PLAY_ROW or feet > 224:
                    continue
                img = env.screen[feet - SPRITE_H:feet, col:col + SPRITE_W]
                world = screen_left(ram) + CROP + col
                behind = panoramas[level][feet - SPRITE_H - PLAY_ROW:feet - PLAY_ROW, world:world + SPRITE_W]
                if behind.shape != img.shape:
                    continue
                mask = ((np.abs(img[:, :, None, :].astype(int) - MARIO_PALETTE[None, None].astype(int))
                         .sum(axis=3).min(axis=2) == 0) & np.any(img != behind, axis=2))
                rows, cols = np.nonzero(mask)
                if len(rows) < 40 or rows.max() - rows.min() < 11:
                    continue
                r0, r1, c0, c1 = rows.min(), rows.max() + 1, cols.min(), cols.max() + 1
                pose = (img[r0:r1, c0:c1] * mask[r0:r1, c0:c1, None], mask[r0:r1, c0:c1], 2 - c0, SPRITE_H - r0)
                key = pose[0].tobytes() + pose[1].tobytes()
                counts[key] = counts.get(key, 0) + 1
                poses.setdefault(key, pose)
    sprites = [poses[k] for k, n in counts.items() if n >= 2]
    # Fire Mario: same poses, red -> pale, brown -> red. (Setting the power-up
    # in RAM doesn't change his palette, so they can't be recorded directly.)
    red, brown, pale = np.array(MARIO_COLORS[0]), np.array(MARIO_COLORS[2]), np.array([252, 216, 168])
    fire = []
    for img, mask, dx, df in sprites:
        f = img.copy()
        f[np.all(img == red, axis=2)] = pale
        f[np.all(img == brown, axis=2)] = red
        fire.append((f, mask, dx, df))
    return sprites + fire


def is_playing(screen):
    """Gameplay rather than the black WORLD x-y card: the bottom rows (ground) aren't black."""
    return screen[-32:].mean() > 12


class ColorFit:
    """Our emulator's colors -> this video's, an affine RGB map fitted over every
    frame so far (a capture's color shift is the same throughout; one dark
    night-level frame alone would fit it badly)."""

    def __init__(self):
        self.xtx = np.eye(4) * 1e-3
        self.xty = np.zeros((4, 3))
        self.xty[:3] = np.eye(3) * 1e-3

    def add(self, ours, theirs):
        a = np.c_[ours.reshape(-1, 3), np.ones(ours.shape[0] * ours.shape[1])][::7]
        self.xtx += a.T @ a
        self.xty += a.T @ theirs.reshape(-1, 3)[::7]

    def __call__(self, colors):
        fit = np.linalg.solve(self.xtx, self.xty)
        return np.c_[colors.reshape(-1, 3), np.ones(colors.size // 3)] @ fit


class Tracker:
    """Follows Mario through one level's main area."""

    def __init__(self, pano, colors, sprites):
        self.colors = colors
        self.sprites = sprites
        # The panorama was stitched from a run, so it has that run's Mario
        # standing at the start: paint him out.
        x, feet = START
        hole = np.zeros(pano.shape[:2], np.uint8)
        hole[feet - 20:feet + 2, x - 4:x + 20] = 1
        pano = cv2.inpaint(np.ascontiguousarray(pano), hole, 3, cv2.INPAINT_TELEA)
        self.pano = pano.astype(np.float32)
        self.pano_gray = cv2.cvtColor(pano, cv2.COLOR_RGB2GRAY).astype(np.float32)
        self.left = 0           # world x of screen column 0
        self.mario = START      # (x, feet row) in panorama coordinates
        self.velocity = (0, 0)
        self.lost = 0

    def scroll(self, play):
        """Find the screen's world x by matching the playfield against the panorama."""
        g = cv2.cvtColor(play, cv2.COLOR_RGB2GRAY).astype(np.float32)
        lo = max(0, self.left + SEARCH[0])
        hi = min(self.pano.shape[1], self.left + SEARCH[1] + g.shape[1])
        if hi - lo < g.shape[1]:
            return 0.0
        res = cv2.matchTemplate(self.pano_gray[:, lo:hi], g, cv2.TM_CCOEFF_NORMED)
        i = int(np.argmax(res))
        self.left = lo + i
        view = self.pano[:, self.left:self.left + g.shape[1]]
        self.colors.add(view, play.astype(np.float32))
        return float(res[0, i])

    def find_mario(self, screen):
        """Mario's (x, feet row) in panorama coordinates, or None.

        Searches the whole screen, status bar included: high jumps (onto the
        flagpole, say) take him above the playfield.
        """
        steps = self.lost + 1
        gx = self.mario[0] + self.velocity[0] * steps - self.left              # expected screen column of his x
        gf = self.mario[1] + self.velocity[1] * steps + PLAY_ROW               # expected screen row of his feet
        reach = int(REACH + REACH_PER_LOST * self.lost)
        c0, c1 = max(0, int(gx) - 2 - reach), min(screen.shape[1], int(gx) - 2 + SPRITE_W + reach)
        r0, r1 = max(0, int(gf) - SPRITE_H - reach), min(screen.shape[0], int(gf) + reach)
        if c1 - c0 < SPRITE_W or r1 - r0 < SPRITE_H:
            self.lost += 1
            return None
        region = screen[r0:r1, c0:c1].astype(np.float32)
        best = None
        for img, mask, dx, df in self.sprites:
            h, w = mask.shape
            if h > region.shape[0] or w > region.shape[1]:
                continue
            tinted = self.colors(img.astype(np.float32)).reshape(img.shape).astype(np.float32)
            m = np.repeat(mask[:, :, None], 3, axis=2).astype(np.float32)
            res = cv2.matchTemplate(region, tinted, cv2.TM_SQDIFF, mask=m) / (3 * mask.sum())
            row, col = np.unravel_index(int(np.argmin(res)), res.shape)
            if best is None or res[row, col] < best[0]:
                best = (res[row, col], (row + df, col + dx))
        if best is None or best[0] > MAX_RMS ** 2:
            self.lost += 1
            return None
        feet, col = best[1]
        x, feet = self.left + c0 + col, r0 + feet - PLAY_ROW
        self.velocity = ((x - self.mario[0]) / steps, (feet - self.mario[1]) / steps)
        self.mario, self.lost = (x, feet), 0
        return self.mario


def track(video, crop, panoramas, glyphs, sprites):
    """{level: [(nes frame, x, y) ...]} for each of our levels seen in the video."""
    reader = imageio.get_reader(video)
    fps = reader.get_meta_data()["fps"]
    paths, level, tracker, t0 = {}, None, None, None
    colors = ColorFit()
    for i, frame in enumerate(reader):
        screen = nes_screen(frame, crop)
        shown = read_level(screen, glyphs)
        if shown != level or not is_playing(screen):
            if shown != level:
                level, tracker, t0 = shown, None, None
            continue
        if level not in panoramas:
            continue
        if tracker is None:
            tracker, t0 = Tracker(panoramas[level], colors, sprites), i
            paths[level] = []
        nes_frame = int(round((i - t0) / fps * NES_FPS))
        play = screen[PLAY_ROW:]
        if tracker.scroll(play) < MIN_SCROLL_MATCH:
            paths[level].append((nes_frame, None, None))
            continue
        m = tracker.find_mario(screen)
        # Same units as the episodes: x = world x of Mario's left edge, y = page*256 + Player_Y
        # (his feet are 256 px below y in panorama rows).
        paths[level].append((nes_frame, m[0], m[1] + 256) if m else (nes_frame, None, None))
    return paths


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True)
    p.add_argument("--crop", help="X,Y,W,H of the 256x240 NES picture in the video")
    p.add_argument("--preview", type=int, help="save this video frame with a grid, to pick --crop")
    p.add_argument("--run", default="main")
    p.add_argument("--source", default="human world record", help="label shown in videos")
    p.add_argument("--out-dir", help="default runs/<run>/human")
    args = p.parse_args()

    if args.preview is not None:
        frame = imageio.get_reader(args.video).get_data(args.preview).copy()
        for x in range(0, frame.shape[1], 50):
            cv2.line(frame, (x, 0), (x, frame.shape[0]), (255, 0, 255) if x % 250 else (0, 255, 255), 1)
        for y in range(0, frame.shape[0], 50):
            cv2.line(frame, (0, y), (frame.shape[1], y), (255, 0, 255) if y % 250 else (0, 255, 255), 1)
        out = os.path.join("videos", "other", f"preview_{args.preview}.png")
        imageio.imwrite(out, frame)
        print("wrote", out, frame.shape)
        return

    crop = tuple(int(v) for v in args.crop.split(","))
    run_dir = os.path.join("runs", args.run)
    records = {}
    for line in open(os.path.join(run_dir, "records.jsonl")):
        r = json.loads(line)
        if r.get("actions") and (r["level"] not in records or r["frames"] < records[r["level"]]["frames"]):
            records[r["level"]] = r
    env = SMBEnv(LEVELS, rank=94)
    glyphs = glyph_library(env)
    # Our record runs reach the flag, so their panoramas cover the whole level;
    # the flag is where our record's last step ended.
    panoramas = {l: build_panorama(env, l, r["actions"]) for l, r in records.items()}
    sprites = mario_sprites(env, [(l, records[l]["actions"]) for l in ("1-1", "8-1", "6-3") if l in records], panoramas)
    print(f"{len(sprites)} Mario sprites", flush=True)
    env.close()

    flag_x = {}
    env = SMBEnv(LEVELS, rank=94)
    for level, r in records.items():
        env.reset(options={"level": level})
        for a in r["actions"]:
            _, _, term, trunc, info = env.step(int(a))
            if term or trunc:
                break
        flag_x[level] = info["x"]
    env.close()

    out_dir = args.out_dir or os.path.join(run_dir, "human")
    os.makedirs(out_dir, exist_ok=True)
    for level, path in track(args.video, crop, panoramas, glyphs, sprites).items():
        found = [(f, x, y) for f, x, y in path if x is not None]
        flag = next((f for f, x, _ in found if x >= flag_x[level] - 4), None)
        n = max(f for f, _, _ in path) + 1
        xs, ys = [None] * n, [None] * n
        for f, x, y in found:
            xs[f], ys[f] = int(x), int(y)
        json.dump({"level": level, "source": args.source, "flag_frame": flag, "x": xs, "y": ys},
                  open(os.path.join(out_dir, f"{level}.json"), "w"))
        print(f"{level}: {len(found)}/{len(path)} frames tracked, "
              f"flag {'at %.2fs' % (flag / NES_FPS) if flag else 'not reached'}")


if __name__ == "__main__":
    main()
