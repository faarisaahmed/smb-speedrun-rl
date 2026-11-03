"""Play a published TAS (an FCEUX .fm2 movie) and save Mario's path through our levels.

The ghost racing the agent in the videos. A TAS is exact inputs, so this plays
it from power-on in our own emulator and reads Mario's position straight from
RAM, timed the same way as our episodes:

  python track_tas.py --movie path/to/movie.fm2 --source "HappyLee & Mars608 TAS"

Writes runs/<run>/tas/<level>.json: {level, source, flag_frame, x, y, big}, where
x[f], y[f] are Mario's x and y (as in our episodes) f frames after the level's
start (the frame he becomes controllable, as in our start states), or None
outside the level's main area, and big[f] whether he's big (or fire).
next_start is the frame the following level becomes controllable: what a
full run is really timed by (after the flag come fireworks, the castle walk
and the wait for the next 21-frame rule). Also saves <level>.state, the
emulator at the level's start, for racing the agent from the TAS's situation
(race_tas.py). flag_frame is the first frame on the flagpole.
"""
import argparse
import gzip
import json
import os

import numpy as np
import stable_retro as retro

from make_states import OUTDOOR_LEVELS
from smb.env import area_id
from smb.ram import RAM, x_pos, y_pos

# FCEUX's fm2 button columns, in order.
FM2_BUTTONS = {"R": "RIGHT", "L": "LEFT", "D": "DOWN", "U": "UP", "T": "START", "S": "SELECT", "B": "B", "A": "A"}
LOADING, NORMAL = 0x07, 0x08
# stable-retro's core boots 3 frames ahead of FCEUX: without dropping them, the
# Start press lands one 21-frame rule later, the game's random numbers differ
# and the movie desyncs (2-2's Bloopers). With them dropped it plays to the end.
BOOT_OFFSET = 3


def read_fm2(path):
    """Per-frame lists of held buttons for player 1."""
    frames = []
    for line in open(path):
        if line.startswith("|"):
            pad = line.split("|")[2]
            frames.append([FM2_BUTTONS[c] for c, k in zip("RLDUTSBA", pad) if k not in ". "])
    return frames


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--movie", required=True)
    p.add_argument("--source", default="TAS")
    p.add_argument("--run", default="main")
    p.add_argument("--levels", nargs="*", default=OUTDOOR_LEVELS)
    args = p.parse_args()

    paths = {}
    cur = {"level": None, "t0": None, "area": None, "prev": None, "i": -1}

    def record(ram):
        """Returns the level whose start this frame is, if it's one of ours."""
        cur["i"] += 1
        state = ram[RAM.PLAYER_STATE]
        shown = f"{ram[RAM.WORLD] + 1}-{ram[RAM.STAGE] + 1}"
        started = None
        if ram[RAM.GAME_MODE] == 1 and shown != cur["level"] and cur["prev"] == LOADING and state == NORMAL:
            # A new level's first controllable frame: where our start states are saved.
            if cur["level"] in paths:
                paths[cur["level"]]["next_start"] = cur["i"] - cur["t0"]
            cur.update(level=shown, t0=cur["i"], area=area_id(ram))
            if shown in args.levels and shown not in paths:
                paths[shown] = {"x": [], "y": [], "big": [], "flag_frame": None, "next_start": None}
                started = shown
                print(f"{shown} starts", flush=True)
        cur["prev"] = state
        path = paths.get(cur["level"])
        if path is None or path["flag_frame"] is not None:
            return started
        main = area_id(ram) == cur["area"]
        path["x"].append(int(x_pos(ram)) if main else None)
        path["y"].append(int(y_pos(ram)) if main else None)
        path["big"].append(bool(ram[RAM.PLAYER_SIZE] == 0) if main else None)
        if ram[RAM.FLOAT_STATE] == RAM.FLAGPOLE:
            path["flag_frame"] = cur["i"] - cur["t0"]
        return started

    # A fresh emulator powered on (reset() without a state doesn't fully power-cycle).
    env = retro.make("SuperMarioBros-Nes-v0", state=retro.State.NONE,
                     use_restricted_actions=retro.Actions.ALL, render_mode=None)
    env.reset()
    names = env.unwrapped.buttons
    out = os.path.join("runs", args.run, "tas")
    os.makedirs(out, exist_ok=True)
    for held in read_fm2(args.movie)[BOOT_OFFSET:]:
        env.step(np.array([b in held for b in names], np.int8))
        started = record(env.unwrapped.get_ram())
        if started:
            with gzip.open(os.path.join(out, f"{started}.state"), "wb") as fh:
                fh.write(env.unwrapped.em.get_state())
    env.close()

    for level in args.levels:
        if level not in paths:
            print(f"{level}: not in the movie")
            continue
        path = paths[level]
        json.dump({"level": level, "source": args.source, **path}, open(os.path.join(out, f"{level}.json"), "w"))
        flag = path["flag_frame"]
        print(f"{level}: flag {'at %.2fs' % (flag / 60.0988) if flag is not None else 'not reached'}, "
              f"next level at {path['next_start']}")


if __name__ == "__main__":
    main()
