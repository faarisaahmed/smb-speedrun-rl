"""Hints from a TAS: practice starts where it takes a shortcut, as small Mario.

Self-play rarely finds that a pipe is worth going down (the 6-2 pipe, say),
because the payoff only shows up long after. This plays the TAS and, at each
point where it leaves a level's main area or comes back into it, or skips
ahead by a glitch, saves the emulator a few moments before and once Mario is
controllable after it, made small (the agent plays power-up-less). Training restarts from
these like any other frontier point (frontier/<level>/hint/, see HINT_DIR in
smb/env.py), so the agent learns that the place exists and pays off; the
inputs that get it there and through are still its own.

  python make_hints.py --movie videos/source/happylee_mars608-smb-warpless.fm2
"""
import argparse
import collections
import gzip
import os
import shutil

import numpy as np
import stable_retro as retro

from make_states import OUTDOOR_LEVELS
from smb.env import AREA_KEY, FRONTIER_BIN, HINT_DIR, area_id
from smb.ram import RAM, x_pos
from track_tas import BOOT_OFFSET, LOADING, NORMAL, read_fm2

BEFORE = (150, 90, 45, 20, 10)   # frames before the TAS's last controllable moment in an area
AFTER = (0, 60)     # frames after it's controllable in the next one
SKIP_DX = 32        # px in one frame: a glitch (Mario runs at most ~3.5 px a frame)


def small(env, state):
    """The same emulator state with Mario small and without power-ups."""
    env.unwrapped.em.set_state(state)
    mem = env.unwrapped.data.memory
    mem.assign(RAM.PLAYER_SIZE, "|u1", 1)
    mem.assign(RAM.POWERUP, "|u1", 0)
    return env.unwrapped.em.get_state()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--movie", required=True)
    p.add_argument("--run", default="main")
    p.add_argument("--levels", nargs="*", default=OUTDOOR_LEVELS)
    args = p.parse_args()

    env = retro.make("SuperMarioBros-Nes-v0", state=retro.State.NONE,
                     use_restricted_actions=retro.Actions.ALL, render_mode=None)
    env.reset()
    names = env.unwrapped.buttons
    recent = collections.deque(maxlen=max(BEFORE) + 1)
    level, area, prev, last_x = None, None, None, 0
    mains = {}            # level -> its main area
    pending = []          # (frame to save at, level) after entering an area
    hints = collections.defaultdict(list)   # level -> raw states
    for i, held in enumerate(read_fm2(args.movie)[BOOT_OFFSET:]):
        env.step(np.array([b in held for b in names], np.int8))
        ram = env.unwrapped.get_ram()
        state = ram[RAM.PLAYER_STATE]
        if ram[RAM.GAME_MODE] == 1 and prev == LOADING and state == NORMAL:
            shown = f"{ram[RAM.WORLD] + 1}-{ram[RAM.STAGE] + 1}"
            if shown != level:
                level = shown
                mains.setdefault(level, area_id(ram))
            elif level in args.levels and area_id(ram) != area:
                # Controllable in another area (down a pipe, up a vine): hints
                # from a moment before the TAS left, and from here.
                if len(recent) > max(BEFORE):
                    hints[level] += [recent[-1 - f] for f in BEFORE]
                pending += [(i + f, level) for f in AFTER]
                print(f"{level}: TAS goes from area {area} to {area_id(ram)}", flush=True)
            recent.clear()
            area = area_id(ram)
        if level in args.levels and ram[RAM.GAME_MODE] == 1 and state == NORMAL and area_id(ram) == area:
            if prev == NORMAL and abs(x_pos(ram) - last_x) >= SKIP_DX and len(recent) > max(BEFORE):
                # Mario skipped ahead within the area: a glitch (6-2's vine).
                hints[level] += [recent[-1 - f] for f in BEFORE]
                pending += [(i + f, level) for f in AFTER]
                print(f"{level}: TAS skips from x={last_x} to {x_pos(ram)}", flush=True)
            recent.append(env.unwrapped.em.get_state())
        last_x = x_pos(ram)
        for f, lv in [q for q in pending if q[0] == i]:
            hints[lv].append(env.unwrapped.em.get_state())
        prev = state

    for lv, states in hints.items():
        out = os.path.join("runs", args.run, "frontier", lv, HINT_DIR)
        shutil.rmtree(out, ignore_errors=True)
        os.makedirs(out)
        for n, s in enumerate(states):
            s = small(env, s)
            env.unwrapped.em.set_state(s)
            ram = env.unwrapped.get_ram()
            # Named like explorer demos: frontier key * 1000 + index (see SMBEnv.start_bin).
            a = area_id(ram)
            key = (0 if a == mains[lv] else a + 1) * AREA_KEY + x_pos(ram) // FRONTIER_BIN
            with gzip.open(os.path.join(out, f"{key * 1000 + n}.state"), "wb") as fh:
                fh.write(s)
            print(f"{lv}: hint {n} at area {a} x={x_pos(ram)}")
    env.close()


if __name__ == "__main__":
    main()
