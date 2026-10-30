"""Find a way past walls the policy keeps dying at (Go-Explore's exploration phase).

For each restart point the policy reliably dies after at the same spot (from
the recent training episodes), try random button sequences, each button held
for a random stretch, until one gets TARGET_MARGIN px further and lands on
solid ground. Of up to KEEP_BEST successes the shortest is kept: snapshots
along it are saved as demo restart points (frontier/<level>/demo/<bin><step>.state),
and its observations and inputs as demo/<bin>.npz, which training imitates
alongside PPO (self-imitation, see smb/sil.py). The policy still has to fit
the move into its own play; nothing is hard-coded into its controls.

  python explore.py --levels 1-3
"""
import argparse
import collections
import glob
import gzip
import json
import os
import time

import numpy as np

from smb.env import ACTIONS, DEMO_DIR, FRONTIER_BIN, SMBEnv
from smb.ram import RAM, x_pos

TARGET_MARGIN = 64   # px past where the policy usually dies
WALL_RATE = 0.05     # a start is "walled" if fewer than this share of attempts get past its target
MIN_ATTEMPTS = 50
REPEAT_PROB = 0.9    # chance of holding the previous button combo another step
MAX_STEPS = 150
SAVE_EVERY = 3       # steps between demo snapshots
KEEP_BEST = 10       # successes to collect before keeping the shortest


def recent_reach(run_dir, level, hours):
    """{start bin: [max_x of each recent mid-level attempt]} plus flag counts."""
    since = time.time() - hours * 3600
    reach, flags = collections.defaultdict(list), collections.Counter()
    for path in glob.glob(os.path.join(run_dir, "episodes", "*.npz")):
        if os.path.getmtime(path) < since:
            continue
        for m in json.loads(str(np.load(path)["meta"])):
            if m["level"] == level and not m.get("from_start", True) and m["time"] >= since:
                b = m["start_x"] // FRONTIER_BIN
                reach[b].append(m["max_x"])
                flags[b] += m["outcome"] == "flag"
    return reach, flags


def explore(env, level, state, target, tries, rng):
    """Random sticky-input runs from state.

    Returns the shortest successful run as (states, observations, actions),
    where observations[i] is what the agent saw before actions[i], or None.
    """
    best = None
    successes = 0
    for _ in range(tries):
        obs, _ = env.reset(options={"level": level, "state": state})
        states, observations, actions = [env.emu.em.get_state()], [], []
        a = int(rng.integers(len(ACTIONS)))
        for _ in range(MAX_STEPS):
            if rng.random() > REPEAT_PROB:
                a = int(rng.integers(len(ACTIONS)))
            observations.append(obs)
            actions.append(a)
            obs, _, term, trunc, _ = env.step(a)
            if term or trunc:
                break
            states.append(env.emu.em.get_state())
            ram = env.emu.get_ram()
            if x_pos(ram) >= target and ram[RAM.FLOAT_STATE] == 0 and ram[RAM.Y_PAGE] == 1:
                if best is None or len(actions) < len(best[2]):
                    best = (states, observations, actions)
                successes += 1
                break
        if successes >= KEEP_BEST:
            break
    return best


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", default="main")
    p.add_argument("--levels", nargs="+", required=True)
    p.add_argument("--hours", type=float, default=1.0, help="how far back to look at training episodes")
    p.add_argument("--tries", type=int, default=5000)
    args = p.parse_args()
    run_dir = os.path.join("runs", args.run)
    rng = np.random.default_rng()
    env = SMBEnv(args.levels, rank=98)

    for level in args.levels:
        d = os.path.join(run_dir, "frontier", level)
        reach, flags = recent_reach(run_dir, level, args.hours)
        walled = []
        for f in os.listdir(d) if os.path.isdir(d) else []:
            if not f.endswith(".state") or f.startswith("."):
                continue
            b = int(f.split(".")[0])
            xs = np.array(reach.get(b, []))
            if len(xs) < MIN_ATTEMPTS or flags[b] / len(xs) >= 0.5:
                continue
            target = int(np.median(xs)) + TARGET_MARGIN
            if np.mean(xs >= target) < WALL_RATE:
                walled.append((b, target, os.path.join(d, f)))
        # Furthest starts first: they're closest to their wall, and one success
        # covers the earlier starts that die at the same wall.
        walled.sort(key=lambda w: -w[0])
        cleared = set()
        out = os.path.join(d, DEMO_DIR)
        have = os.listdir(out) if os.path.isdir(out) else []
        for b, target, path in walled:
            if any(abs(target - t) <= FRONTIER_BIN // 4 for t in cleared):
                continue
            if any(f.startswith(f"{b:03d}") for f in have):
                cleared.add(target)   # already has a demo; training is still learning it
                continue
            with gzip.open(path, "rb") as fh:
                state = fh.read()
            t0 = time.time()
            found = explore(env, level, state, target, args.tries, rng)
            if found is None:
                print(f"{level} bin {b}: nothing reached x={target} in {args.tries} tries", flush=True)
                continue
            cleared.add(target)
            states, observations, actions = found
            os.makedirs(out, exist_ok=True)
            np.savez_compressed(os.path.join(out, f"{b:03d}.npz"), actions=np.array(actions, np.uint8),
                                **{k: np.stack([o[k] for o in observations]) for k in observations[0]})
            for step in range(0, len(states), SAVE_EVERY):
                with gzip.open(os.path.join(out, f"{b:03d}{step:03d}.state"), "wb") as fh:
                    fh.write(states[step])
            print(f"{level} bin {b}: reached x={target} ({len(states)} steps, {time.time() - t0:.0f}s), "
                  f"saved {len(range(0, len(states), SAVE_EVERY))} demo points", flush=True)
    env.close()


if __name__ == "__main__":
    main()
