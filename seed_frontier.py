"""Seed frontier restart points from the agent's own furthest recorded run.

Replays the furthest full-level attempt of each level and saves a snapshot
every FRONTIER_BIN px (except the last SAFE_STEPS before it died) to
runs/<run>/frontier/<level>/<bin>.state, which training envs load.
With --demo, instead saves a snapshot every --every steps, mid-air included,
to runs/<run>/frontier/<level>/demo/<step>.state (see DEMO_DIR in smb/env.py).

  python seed_frontier.py --levels 1-3
  python seed_frontier.py --levels 1-3 --demo
"""
import argparse
import gzip
import os

from ghosts import load_episodes
from smb.env import AREA_CHANGE_DX, DEMO_DIR, FRONTIER_BIN, SAFE_STEPS, SMBEnv


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", default="main")
    p.add_argument("--levels", nargs="+", required=True)
    p.add_argument("--demo", action="store_true")
    p.add_argument("--every", type=int, default=3)
    args = p.parse_args()
    run_dir = os.path.join("runs", args.run)

    env = SMBEnv(args.levels, rank=95)
    for level in args.levels:
        episodes = load_episodes(run_dir, level)
        if not episodes:
            print(level, "no episodes")
            continue
        meta, actions, xs, _ = max(episodes, key=lambda ep: ep[0]["max_x"])
        last_safe = len(actions) - SAFE_STEPS if meta["outcome"] == "death" else len(actions)
        out = os.path.join(run_dir, "frontier", level, *([DEMO_DIR] if args.demo else []))
        os.makedirs(out, exist_ok=True)
        env.reset(options={"level": level})
        next_bin, prev_x, saved = env.max_x // FRONTIER_BIN + 1, env.max_x, []
        for step, a in enumerate(actions[:last_safe], 1):
            env.step(int(a))
            x = env.prev_x
            if abs(x - prev_x) >= AREA_CHANGE_DX:
                break
            prev_x = x
            if args.demo and step % args.every == 0:
                with gzip.open(os.path.join(out, f"{step}.state"), "wb") as f:
                    f.write(env.emu.em.get_state())
                saved.append(x)
            elif not args.demo and x // FRONTIER_BIN >= next_bin:
                next_bin = x // FRONTIER_BIN + 1
                with gzip.open(os.path.join(out, f"{x // FRONTIER_BIN}.state"), "wb") as f:
                    f.write(env.emu.em.get_state())
                saved.append(x)
        print(f"{level}: seeded {len(saved)} points up to x={max(saved, default=0)} "
              f"from a run that reached x={meta['max_x']}")
    env.close()


if __name__ == "__main__":
    main()
