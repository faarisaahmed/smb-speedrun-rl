"""Build a save state at the start of each outdoor level.

Boots the game, presses Start, and keeps overwriting the world/level/area
RAM bytes until the level loads, then saves the state the moment Mario
becomes controllable. Writes states/<W>-<L>.state and a preview PNG.
"""
import argparse
import gzip
import os

import cv2
import numpy as np
import stable_retro as retro

from smb.ram import RAM

# Outdoor levels only: no underground (1-2, 4-2), no underwater (2-2, 7-2), no castles (X-4).
OUTDOOR_LEVELS = [
    "1-1", "1-3", "2-1", "2-3", "3-1", "3-2", "3-3", "4-1", "4-3",
    "5-1", "5-2", "5-3", "6-1", "6-2", "6-3", "7-1", "7-3", "8-1", "8-2", "8-3",
]


def area_for(world, stage):
    # Worlds 1, 2, 4 and 7 have a pipe-entry cutscene area before stage 2,
    # which shifts the area index of every later stage by one.
    if world in (1, 2, 4, 7) and stage >= 2:
        return stage + 1
    return stage


def make_state(env, world, stage):
    mem = env.unwrapped.data.memory
    env.reset()
    noop = np.zeros(9, dtype=np.int8)
    start = noop.copy()
    start[env.unwrapped.buttons.index("START")] = 1

    def write_stage():
        mem.assign(RAM.WORLD, "|u1", world - 1)
        mem.assign(RAM.STAGE, "|u1", stage - 1)
        mem.assign(RAM.AREA, "|u1", area_for(world, stage) - 1)

    # The title-screen demo leaves PLAYER_STATE at 8, so only accept 8 after
    # the game has started and Mario has gone through the area-entry state.
    in_game = loading = False
    for frame in range(3000):
        ram = env.unwrapped.get_ram()
        if not in_game and ram[RAM.GAME_MODE] == 1:
            in_game = True
        if in_game and ram[RAM.PLAYER_STATE] == 7:
            loading = True
        if loading and ram[RAM.PLAYER_STATE] == 8:
            break
        write_stage()
        env.step(start if (not in_game and frame % 20 == 10) else noop)
    else:
        raise RuntimeError(f"level {world}-{stage} never became playable")

    ram = env.unwrapped.get_ram()
    assert ram[RAM.WORLD] == world - 1 and ram[RAM.STAGE] == stage - 1, (world, stage)
    return env.unwrapped.em.get_state(), env.unwrapped.get_screen()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--levels", nargs="*", default=OUTDOOR_LEVELS)
    p.add_argument("--out", default="states")
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)

    for name in args.levels:
        world, stage = map(int, name.split("-"))
        # Fresh emulator per level: reset() without a state doesn't fully power-cycle.
        env = retro.make("SuperMarioBros-Nes-v0", state=retro.State.NONE,
                         use_restricted_actions=retro.Actions.ALL, render_mode=None)
        state, screen = make_state(env, world, stage)
        env.close()
        with gzip.open(os.path.join(args.out, f"{name}.state"), "wb") as f:
            f.write(state)
        cv2.imwrite(os.path.join(args.out, f"{name}.png"), cv2.cvtColor(screen, cv2.COLOR_RGB2BGR))
        print("saved", name)


if __name__ == "__main__":
    main()
