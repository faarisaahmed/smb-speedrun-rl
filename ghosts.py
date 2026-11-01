"""Ghost-overlay video: many recorded attempts at a level played at once.

Takes a batch of training episodes from around one point in training and
draws every attempt as a translucent Mario-sized ghost moving through a
stitched panorama of the level. Ghosts that die leave a red x behind.
The camera follows the leading ghost.

  python ghosts.py --level 1-1 --at 2000000 --count 300
  python ghosts.py --level 1-1 --at latest
  python ghosts.py --level 1-1 --panorama-only     # just save the level image
  python ghosts.py --level 1-1 --checkpoint runs/main/checkpoints/step_0000500000.zip
      # fresh attempts played by a saved policy, for points in training whose
      # recorded episodes are gone

The panorama is stitched from screen columns as they scroll into view, so an
enemy that happened to be at the right edge can get baked in.
"""
import argparse
import glob
import json
import os

import cv2
import imageio
import numpy as np

from smb.env import ACTIONS, SMBEnv
from smb.human import HumanGhost, luigi
from smb.obs import screen_left
from smb.ram import RAM, x_pos
from smb.viz import draw_pad

NES_FPS = 60.0988
# stable-retro's 240x224 NES image crops 8 px off each side and the top, so
# image (col, row) = (world x - screen_left - 8, NES y - 8). The panorama
# starts at NES y 32 (below the status bar) and is indexed by world x.
CROP = 8
TOP = 32
SCREEN_W = 240
SCALE = 3
GHOST_ALPHA = 0.18
MARIO_COLORS = [(216, 40, 0), (248, 152, 56), (136, 112, 0)]


def load_episodes(run_dir, level):
    """All recorded episodes of a level, as (meta, actions, x, y), oldest first."""
    episodes = []
    for path in sorted(glob.glob(os.path.join(run_dir, "episodes", "*.npz"))):
        data = np.load(path)
        metas = json.loads(str(data["meta"]))
        ends = np.cumsum(data["lengths"])
        starts = ends - data["lengths"]
        for meta, s, e in zip(metas, starts, ends):
            if meta["level"] == level and meta.get("from_start", True):
                episodes.append((meta, data["actions"][s:e], data["x"][s:e].astype(int), data["y"][s:e].astype(int)))
    episodes.sort(key=lambda ep: ep[0]["time"])
    return episodes


def step_clock(run_dir):
    """Map wall-clock time to global training step.

    Per-env step counters restart when training resumes, so use the logs
    that record both a time and the global step instead.
    """
    points = []
    for name in ("records.jsonl", "evals.jsonl"):
        path = os.path.join(run_dir, name)
        if os.path.exists(path):
            for line in open(path):
                r = json.loads(line)
                points.append((r["time"], r["timesteps"]))
    d = os.path.join(run_dir, "checkpoints")
    for f in os.listdir(d) if os.path.isdir(d) else []:
        points.append((os.path.getmtime(os.path.join(d, f)), int(f[len("step_"):-len(".zip")])))
    points.sort()
    times, steps = zip(*points) if points else ((0, 1), (0, 0))
    return lambda t: float(np.interp(t, times, steps))


def mario_sprite(env):
    """Small Mario cut out of the 1-1 start screen: (BGR pixels, mask)."""
    env.reset(options={"level": "1-1"})
    ram = env.emu.get_ram()
    col = x_pos(ram) - screen_left(ram) - CROP
    row = int(ram[RAM.Y_SCREEN]) + 16 - CROP
    patch = env.screen[row:row + 16, col:col + 16]
    mask = np.zeros(patch.shape[:2], bool)
    for color in MARIO_COLORS:   # Mario starts in front of the hill, so mask by his own palette
        mask |= np.all(patch == color, axis=2)
    return cv2.cvtColor(np.ascontiguousarray(patch), cv2.COLOR_RGB2BGR).astype(np.float32), mask


def draw_sprite(view, sprite, mask, sx, sy, alpha):
    h, w = mask.shape
    x0, y0 = max(sx, 0), max(sy, 0)
    x1, y1 = min(sx + w, view.shape[1]), min(sy + h, view.shape[0])
    if x0 >= x1 or y0 >= y1:
        return
    m = mask[y0 - sy:y1 - sy, x0 - sx:x1 - sx]
    region = view[y0:y1, x0:x1]
    region[m] = region[m] * (1 - alpha) + sprite[y0 - sy:y1 - sy, x0 - sx:x1 - sx][m] * alpha


def play_checkpoint(path, level, count):
    """count attempts from the level start by a saved policy, sampling its actions as in training."""
    from stable_baselines3 import PPO
    model = PPO.load(path, device="cpu")
    env = SMBEnv([level], rank=95)
    episodes = []
    for _ in range(count):
        obs, _ = env.reset(options={"level": level})
        while True:
            action, _ = model.predict(obs, deterministic=False)
            obs, _, terminated, truncated, info = env.step(int(action))
            if terminated or truncated:
                break
        episodes.append((info["episode_meta"], np.array(env.actions), np.array(env.xs), np.array(env.ys)))
    env.close()
    return episodes


def build_panorama(env, level, actions):
    """Replay a long run and stitch each newly revealed screen column."""
    env.reset(options={"level": level})
    pano = np.zeros((224 - (TOP - CROP), 32 * 256, 3), np.uint8)   # 8-1 is over 16 screens long
    filled = 0
    for a in actions:
        _, _, term, trunc, _ = env.step(int(a))
        left = screen_left(env.emu.get_ram()) + CROP   # world x of image column 0
        right = left + SCREEN_W
        if right > filled:
            new_from = max(filled, left)
            pano[:, new_from:right] = env.screen[TOP - CROP:, new_from - left:SCREEN_W]
            filled = right
        if term or trunc:
            break
    return pano[:, :filled]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", default="main")
    p.add_argument("--level", default="1-1")
    p.add_argument("--at", default="latest", help="global training step to sample around, or 'latest'/'first'")
    p.add_argument("--count", type=int, default=300)
    p.add_argument("--out")
    p.add_argument("--panorama-only", action="store_true")
    p.add_argument("--checkpoint", help="play --count new attempts with this saved policy instead")
    args = p.parse_args()

    run_dir = os.path.join("runs", args.run)
    episodes = load_episodes(run_dir, args.level)
    played = play_checkpoint(args.checkpoint, args.level, args.count) if args.checkpoint else []
    episodes = episodes or played
    if not episodes:
        print("no recorded episodes for", args.level, "yet (they're flushed every 200 per env)")
        return

    # Panorama from the run that got furthest.
    env = SMBEnv([args.level], rank=96)
    furthest = max(episodes, key=lambda ep: ep[0]["max_x"])
    pano = build_panorama(env, args.level, furthest[1])
    sprite, mask = mario_sprite(env)
    luigi_sprite, _ = luigi(sprite, mask)
    human = HumanGhost(run_dir, args.level)
    env.close()
    clock = step_clock(run_dir)
    for folder in ("other", "ghosts"):
        os.makedirs(os.path.join("videos", folder), exist_ok=True)
    pano_path = os.path.join("videos", "other", f"panorama_{args.level}.png")
    cv2.imwrite(pano_path, cv2.cvtColor(pano, cv2.COLOR_RGB2BGR))
    if args.panorama_only:
        print("wrote", pano_path)
        return

    if played:
        batch = played
    elif args.at == "first":
        batch = episodes[:args.count]
    elif args.at == "latest":
        batch = episodes[-args.count:]
    else:
        target = int(args.at)
        i = int(np.argmin([abs(clock(ep[0]["time"]) - target) for ep in episodes]))
        lo = max(0, min(i - args.count // 2, len(episodes) - args.count))
        batch = episodes[lo:lo + args.count]
    if played:
        approx = int(os.path.basename(args.checkpoint)[len("step_"):-len(".zip")])
    else:
        approx = int(np.mean([clock(ep[0]["time"]) for ep in batch]))
    cleared = sum(ep[0]["outcome"] == "flag" for ep in batch)

    out = args.out or os.path.join("videos", "ghosts", f"{args.level}_{args.at}.mp4")
    writer = imageio.get_writer(out, fps=NES_FPS / batch[0][0]["frame_skip"], macro_block_size=1)
    pano_bgr = cv2.cvtColor(pano, cv2.COLOR_RGB2BGR)
    longest = max(len(ep[1]) for ep in batch)
    deaths = []
    cam = 0.0
    for t in range(longest + 30):
        alive = [(ep[2][t], ep[3][t], ep[1][t]) for ep in batch if t < len(ep[1])]
        for ep in batch:
            if len(ep[1]) == t and ep[0]["outcome"] == "death":
                deaths.append((ep[2][-1], ep[3][-1]))
        lead = max((x for x, _, _ in alive), default=cam + 100)
        cam += (min(max(lead - 100, 0), pano.shape[1] - SCREEN_W) - cam) * 0.15
        c = int(cam)

        view = pano_bgr[:, c:c + SCREEN_W].astype(np.float32)
        for x, y, _ in alive:
            draw_sprite(view, sprite, mask, x - c, y - 256 + 16 - TOP, GHOST_ALPHA)
        if alive:
            x, y, held = max(alive)
            draw_sprite(view, sprite, mask, x - c, y - 256 + 16 - TOP, 1.0)
        pos = human.at(t * batch[0][0]["frame_skip"])
        if pos is not None:
            draw_sprite(view, luigi_sprite, mask, pos[0] - c, pos[1] - 256 + 16 - TOP, 1.0)
        view = view.astype(np.uint8)
        for x, y in deaths:
            sx, sy = x - c + 8, min(y - 256 + 24 - TOP, view.shape[0] - 6)
            cv2.drawMarker(view, (sx, sy), (0, 0, 200), cv2.MARKER_TILTED_CROSS, 6, 1)

        big = cv2.resize(view, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_NEAREST)
        text = f"{args.level}   ~{approx:,} training steps   {len(alive)} alive   {cleared}/{len(batch)} cleared"
        cv2.putText(big, text, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 5, cv2.LINE_AA)
        cv2.putText(big, text, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        if human.ok:
            cv2.putText(big, human.label(), (12, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 5, cv2.LINE_AA)
            cv2.putText(big, human.label(), (12, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (60, 200, 60), 2, cv2.LINE_AA)
        if alive:   # the solid (leading) ghost's buttons
            draw_pad(big, 12, big.shape[0] - 106, ACTIONS[held], unit=1.3)
        writer.append_data(big[:, :, ::-1])
    writer.close()
    print(f"wrote {out} ({len(batch)} attempts, {cleared} cleared)")


if __name__ == "__main__":
    main()
