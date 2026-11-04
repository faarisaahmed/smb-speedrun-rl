"""Race the agent against the TAS fairly: from the TAS's own situation at each level.

The agent's records start from our fixed level states (small Mario, one
timing of enemies and platforms), and a level's flag time isn't what a full
run is timed by. Here the latest checkpoint plays from the state the TAS was
in at the level's start (saved by track_tas.py: its powerup, its frame rule,
its enemies), and both are timed to the frame the next level becomes
controllable, which takes in fireworks, the castle walk and the frame rule.

  python race_tas.py --attempts 50

Saves the agent's fastest run per level to runs/<run>/tas_race/<level>.json.
"""
import argparse
import glob
import gzip
import json
import os

import numpy as np

from smb.env import SMBEnv
from smb.ram import RAM
from smb.sil import SILPPO
from track_tas import LOADING, NORMAL

NES_FPS = 60.0988
AFTER_FLAG = 1200   # frames: the castle walk, fireworks and next level load take less


def next_start(env):
    """After the flag, frames until the next level is controllable (no input matters)."""
    raw, prev = env.env, None
    zeros = np.zeros(len(raw.unwrapped.buttons), np.int8)
    for n in range(1, AFTER_FLAG + 1):
        raw.step(zeros)
        ram = raw.unwrapped.get_ram()
        if prev == LOADING and ram[RAM.PLAYER_STATE] == NORMAL:
            return n
        prev = ram[RAM.PLAYER_STATE]
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", default="main")
    p.add_argument("--attempts", type=int, default=50)
    p.add_argument("--levels", nargs="*")
    p.add_argument("--tas", default="tas", help="which tracked TAS to race (a folder from track_tas.py --tas)")
    args = p.parse_args()
    run_dir = os.path.join("runs", args.run)
    tas_dir = os.path.join(run_dir, args.tas)
    out_dir = os.path.join(run_dir, f"{args.tas}_race")
    os.makedirs(out_dir, exist_ok=True)
    ckpt = sorted(glob.glob(os.path.join(run_dir, "checkpoints", "*.zip")))[-1]
    model = SILPPO.load(ckpt, device="cpu")
    levels = args.levels or sorted(os.path.basename(f)[:-len(".state")] for f in glob.glob(os.path.join(tas_dir, "*.state")))
    print(f"agent: {os.path.basename(ckpt)}")
    print(f"{'level':6} {'TAS flag':>9} {'TAS next':>9} {'agent flag':>11} {'agent next':>11} {'cleared':>8}  verdict")
    for level in levels:
        tas = json.load(open(os.path.join(tas_dir, f"{level}.json")))
        with gzip.open(os.path.join(tas_dir, f"{level}.state"), "rb") as fh:
            state = fh.read()
        env = SMBEnv([level], rank=80)
        best, cleared = None, 0
        for attempt in range(args.attempts):
            obs, _ = env.reset(options={"level": level, "state": state})
            actions = []
            while True:
                action, _ = model.predict(obs, deterministic=attempt == 0)
                actions.append(int(action))
                obs, _, terminated, truncated, info = env.step(int(action))
                if terminated or truncated:
                    break
            if info["episode_meta"]["outcome"] != "flag":
                continue
            cleared += 1
            flag = env.frames
            nxt = next_start(env)
            if nxt is not None and (best is None or flag + nxt < best["next_start"]):
                best = {"flag_frame": flag, "next_start": flag + nxt, "actions": actions}
        env.close()
        if best:
            json.dump({"level": level, "checkpoint": os.path.basename(ckpt), **best},
                      open(os.path.join(out_dir, f"{level}.json"), "w"))
        t_flag, t_next = tas["flag_frame"], tas["next_start"]
        if best is None:
            verdict = "agent never cleared"
        elif t_next is None:
            verdict = "?"
        else:
            d = best["next_start"] - t_next
            verdict = "tie" if d == 0 else (f"AGENT by {-d} frames" if d < 0 else f"TAS by {d} frames")
        fmt = lambda f: f"{f / NES_FPS:.2f}s" if f is not None else "--"
        print(f"{level:6} {fmt(t_flag):>9} {fmt(t_next):>9} {fmt(best and best['flag_frame']):>11} "
              f"{fmt(best and best['next_start']):>11} {cleared:>4}/{args.attempts}  {verdict}", flush=True)


if __name__ == "__main__":
    main()
