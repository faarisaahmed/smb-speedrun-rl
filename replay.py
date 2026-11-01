"""Render recorded runs to video by replaying their inputs in the emulator.

  python replay.py records   --level 1-1   # every new fastest time, in order
  python replay.py evals     --level 1-1   # the checkpoint-by-checkpoint progression
  python replay.py anomalies               # glitch candidates (big position jumps)

Options: --run main, --out videos/x.mp4, --plain (game screen only, no
side panel, for editing), --last N (only the N most recent).

  python replay.py records --level 1-3 --best   # just the current record, as
      videos/records/1-3_<time>s.mp4, replacing the level's previous record video
      (training runs this whenever a record falls)
"""
import argparse
import base64
import json
import os

import cv2
import imageio
import numpy as np

from smb.env import ACTIONS, SMBEnv
from smb.obs import observe
from smb.viz import draw_pad, frame

NES_FPS = 60.0988


# Position jumps bigger than a screen are pipe/area transitions (e.g. the
# 1-1 bonus room), not glitches.
AREA_CHANGE_DX = 256


def is_area_change(dx):
    return abs(dx) >= AREA_CHANGE_DX


def load(run_dir, kind, level, include_pipes=False):
    path = os.path.join(run_dir, f"{kind}.jsonl")
    runs = [json.loads(l) for l in open(path)] if os.path.exists(path) else []
    runs = [r for r in runs if r.get("actions") and (level is None or r["level"] == level)]
    if kind == "anomalies" and not include_pipes:
        runs = [r for r in runs if not all(is_area_change(dx) or abs(dx) < 30 for _, dx in r["anomalies"])]
    return runs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("kind", choices=["records", "evals", "anomalies"])
    p.add_argument("--run", default="main")
    p.add_argument("--level")
    p.add_argument("--out")
    p.add_argument("--plain", action="store_true")
    p.add_argument("--last", type=int, default=0)
    p.add_argument("--include-pipes", action="store_true", help="anomalies: keep pipe/area transitions")
    p.add_argument("--best", action="store_true", help="records: only the current record, into videos/records/")
    args = p.parse_args()

    run_dir = os.path.join("runs", args.run)
    runs = load(run_dir, args.kind, args.level, args.include_pipes)
    if args.best:
        runs = [min(runs, key=lambda r: r["frames"])] if runs else []
    if args.last:
        runs = runs[-args.last:]
    if not runs:
        print("nothing to render yet")
        return

    # Records go with the other record videos; glitch candidates with the cool finds.
    folder = {"records": "records", "evals": "other", "anomalies": "cool"}[args.kind]
    out = args.out or os.path.join("videos", folder, f"{args.level or 'all'}_{args.kind}.mp4")
    if args.best:
        out = os.path.join("videos", "records", f"{runs[0]['level']}_{runs[0]['frames'] / NES_FPS:.2f}s.mp4")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    env = SMBEnv([runs[0]["level"]], frame_skip=runs[0]["frame_skip"], rank=97)
    # --best writes to a hidden temporary file first, so the folder never lacks a record.
    tmp = os.path.join(os.path.dirname(out), "." + os.path.basename(out)) if args.best else out
    writer = imageio.get_writer(tmp, fps=NES_FPS / env.frame_skip, macro_block_size=1, format="FFMPEG")

    for i, run in enumerate(runs):
        start = run.get("start_state")
        env.reset(options={"level": run["level"], "state": base64.b64decode(start) if start else None})
        anomaly_steps = {s for s, _ in run.get("anomalies", [])}
        title = {
            "records": f"record #{i + 1}: {run['frames'] / NES_FPS:.2f}s",
            "evals": f"checkpoint {run['timesteps']:,} steps",
            "anomalies": f"glitch candidate {i + 1}/{len(runs)}",
        }[args.kind]
        for step, action in enumerate(run["actions"], 1):
            _, _, terminated, truncated, info = env.step(action)
            if args.plain:
                img = cv2.cvtColor(np.ascontiguousarray(env.screen), cv2.COLOR_RGB2BGR)
                draw_pad(img, 6, img.shape[0] - 46, ACTIONS[action], unit=0.55)
                img = img[:, :, ::-1]
            else:
                tiles, objs, _ = observe(env.emu.get_ram())
                lines = [f"level {run['level']}", title, f"at {run['timesteps']:,} steps",
                         f"time {env.frames / NES_FPS:6.2f}s"]
                if step in anomaly_steps:
                    lines.append("!! position jump")
                img = frame(env.screen, tiles, objs, lines, buttons=ACTIONS[action])[:, :, ::-1]
            writer.append_data(img)
            if terminated or truncated:
                break
        if env.frames != run["frames"]:
            print(f"warning: replay of run {i} diverged ({env.frames} vs {run['frames']} frames)")
        for _ in range(20):
            writer.append_data(img)
        print(f"{title} ({run['outcome']})", flush=True)

    writer.close()
    env.close()
    if args.best:
        os.replace(tmp, out)
        level = runs[0]["level"]
        for f in os.listdir(os.path.dirname(out)):
            path = os.path.join(os.path.dirname(out), f)
            if f.startswith(f"{level}_") and path != out:
                os.remove(path)
    print("wrote", out)


if __name__ == "__main__":
    main()
