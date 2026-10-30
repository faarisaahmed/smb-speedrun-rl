"""Watch the latest checkpoint play, live.

Plays at real NES speed in a window, reloading the newest checkpoint after
every attempt, so you see the agent improve while training runs. Cycles
through the levels training is currently on (plus held-out levels once the
curriculum reaches them).

  python watch.py                    # follow the main run
  python watch.py --level 1-3        # stick to one level
  python watch.py --speed 4          # 4x speed
  python watch.py --video out.mp4 --episodes 3   # write a video instead of a window

Keys in the window: q quit, n next level, space pause.
"""
import argparse
import itertools
import json
import os
import time

import cv2
import imageio
import numpy as np
import torch
from stable_baselines3 import PPO

from smb.env import SMBEnv
from smb.obs import observe
from smb.viz import frame

NES_FPS = 60.0988


def latest_checkpoint(run_dir):
    d = os.path.join(run_dir, "checkpoints")
    ckpts = sorted(f for f in os.listdir(d) if f.endswith(".zip")) if os.path.isdir(d) else []
    return os.path.join(d, ckpts[-1]) if ckpts else None


def read_status(run_dir):
    try:
        return json.load(open(os.path.join(run_dir, "status.json")))
    except (OSError, ValueError):
        return {}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", default="main")
    p.add_argument("--level")
    p.add_argument("--checkpoint", help="specific checkpoint .zip instead of following the latest")
    p.add_argument("--speed", type=float, default=1.0)
    p.add_argument("--stochastic", action="store_true", help="sample actions like training does")
    p.add_argument("--video")
    p.add_argument("--episodes", type=int, default=0, help="stop after N attempts (0 = forever)")
    args = p.parse_args()
    torch.set_num_threads(1)

    run_dir = os.path.join("runs", args.run)
    env = SMBEnv(["1-1"], rank=98)
    writer = imageio.get_writer(args.video, fps=NES_FPS / env.frame_skip, macro_block_size=1) if args.video else None
    if not writer:
        cv2.namedWindow("smb-rl", cv2.WINDOW_AUTOSIZE)

    model, loaded = None, None
    level_cycle = None
    for episode in itertools.count():
        if args.episodes and episode >= args.episodes:
            break
        ckpt = args.checkpoint or latest_checkpoint(run_dir)
        while ckpt is None:
            print("waiting for the first checkpoint...")
            time.sleep(10)
            ckpt = latest_checkpoint(run_dir)
        if ckpt != loaded:
            model = PPO.load(ckpt, device="cpu")
            loaded = ckpt
            status = read_status(run_dir)
            levels = [args.level] if args.level else status.get("levels", ["1-1"])
            if status.get("stage") == 2 and not args.level:
                levels = levels + status.get("holdout", [])
            level_cycle = itertools.cycle(levels)
        level = next(level_cycle)
        status = read_status(run_dir)
        best = status.get("best_frames", {}).get(level)

        obs, _ = env.reset(options={"level": level})
        steps_label = os.path.basename(loaded).replace("step_", "").replace(".zip", "").lstrip("0") or "0"
        held_out = level in status.get("holdout", [])
        paused = False
        while True:
            t0 = time.time()
            action, _ = model.predict(obs, deterministic=not args.stochastic)
            obs, _, terminated, truncated, info = env.step(int(action))
            tiles, objs, _ = observe(env.emu.get_ram())
            lines = [
                f"level {level}" + ("  (never trained on)" if held_out else ""),
                f"training steps {int(steps_label):,}",
                f"attempt {episode + 1}",
                f"time {env.frames / NES_FPS:6.2f}s",
                f"best {best / NES_FPS:.2f}s" if best else "best  --",
            ]
            img = frame(env.screen, tiles, objs, lines)
            if writer:
                writer.append_data(img[:, :, ::-1])
            else:
                cv2.imshow("smb-rl", img)
                delay = max(1, int((env.frame_skip / NES_FPS / args.speed - (time.time() - t0)) * 1000))
                key = cv2.waitKey(delay) & 0xFF
                if key == ord(" "):
                    paused = not paused
                while paused:
                    key = cv2.waitKey(50) & 0xFF
                    if key == ord(" "):
                        paused = False
                    elif key == ord("q"):
                        return
                if key == ord("q"):
                    return
                if key == ord("n"):
                    break
            if terminated or truncated:
                meta = info["episode_meta"]
                print(f"{level}: {meta['outcome']} after {meta['frames'] / NES_FPS:.2f}s "
                      f"(x={meta['max_x']}, checkpoint {steps_label})", flush=True)
                if writer:
                    for _ in range(15):
                        writer.append_data(img[:, :, ::-1])
                break
    if writer:
        writer.close()
    env.close()


if __name__ == "__main__":
    main()
