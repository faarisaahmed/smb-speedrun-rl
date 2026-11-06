# smb-rl

A PPO agent learning to speedrun the real NES **Super Mario Bros.**, running in an
emulator: first 1-1, then 1-3, then every outdoor level, and eventually looking for
speedrun tricks and glitches on its own.

**[Watch it play: faarisaahmed.github.io/smb-speedrun-rl](https://faarisaahmed.github.io/smb-speedrun-rl/)**:
highlights, every record, fair races against tool-assisted runs, and the story of what went wrong.

This is the much better successor to
[super-mario-bros-rl-model](https://github.com/faarisaahmed/super-mario-bros-rl-model).
That project rebuilt the game from scratch in pygame and trained an agent on the
remake. The agent got about 21% of the way through 1-1. This one plays the actual
cartridge and, after 2.38B steps, clears all 17 outdoor training levels as small Mario.

## Results

Best times (power-on of the level to touching the flagpole, small Mario, every run
playable on a real controller):

| level | best | | level | best |
|---|---|---|---|---|
| 1-1 | 20.00 s | | 5-1 | 21.76 s |
| 1-3 | 16.26 s | | 5-2 | 21.36 s |
| 2-1 | 23.43 s | | 6-2 | 26.66 s |
| 2-3 | 24.01 s | | 6-3 | 18.00 s |
| 3-1 | 21.46 s | | 7-1 | 19.35 s |
| 3-2 | 22.30 s | | 7-3 | 24.01 s |
| 3-3 | 16.14 s | | 8-1 | 40.93 s |
| 4-1 | 24.01 s | | 8-3 | 22.85 s |
| 4-3 | 15.71 s | | | |

5-3, 6-1 and 8-2 are never trained on; they're held out to test levels it has never seen.

`race_tas.py` races it fairly against a TAS: the agent plays from the TAS's own save
state at each level's start, and both are timed to the next level's start. See the
site for the results.

## How it works

**Observation:** not pixels. The visible screen is read from RAM as a 13x17 grid of
blocks (terrain tile ids, plus Mario / enemies / platforms / items), along with Mario's
exact position, speed and state, for the current and previous step (`smb/obs.py`).
This lets a dense network (`smb/model.py`) train fast on a CPU.

**Actions:** 12 button combos, chosen every 2 frames. Run is never forced and no
tricks are scripted; the agent has to discover everything itself.

**Reward:** pixels of progress, minus a small cost per step (faster is better), minus
death, plus a flagpole bonus (`smb/env.py`).

**Curriculum** (`train.py`):
1. 1-1
2. 1-1 + 1-3
3. all outdoor levels except the held-out ones

A stage advances once every level in it is cleared in 60% of recent attempts. Weaker
levels get sampled more.

### Getting past walls (the 1-3 story)

1-3 needs a full-height running jump onto a tall tree, from a 3-step take-off window,
holding jump for ~15 steps. PPO alone never got it: the policy's chance of jumping in
that window was ~0, and in the air it let go of A at random, so it never saw a single
success to learn from. Three things fixed it, all general rather than 1-3-specific.

- **Frontier restarts (Go-Explore style):**
  - Snapshots are taken every 64 px Mario reaches while standing on solid ground.
  - Half the episodes restart from one of them.
  - Restart points are weighted toward the ones where attempts fail to get safely
    further, so hard sections get practiced instead of replayed from the start.
- **`explore.py`:** finds restart points where the policy keeps dying at the same spot.
  - It tries random held-button sequences there until one gets past the wall.
  - It keeps the shortest success and saves states along it as extra restart points
    (the "backward" algorithm).
  - Training runs it in the background every 2M steps.
- **Self-imitation (`smb/sil.py`):** after each PPO update, a few small gradient steps
  make the policy more likely to press what the explorer pressed on its successful runs.

Within minutes of adding self-imitation, its chance of jumping in that window went
from ~0 to 97%. 1-3 went from never cleared to 16.66 s.

## Setup

```
uv venv --python 3.10 && uv pip install -r requirements.txt
mkdir roms && cp /path/to/your/smb.nes roms/
.venv/bin/python -m stable_retro.import roms/
.venv/bin/python make_states.py          # start-of-level save states for the outdoor levels
```

Bring your own cartridge dump; no ROM or ROM-derived files are in this repo. The
stable-retro integration expects the standard (headerless SHA-1
`facee9c577a5262dbe33ac4930bb0b58c8c037f7`) dump.

## Commands

```
# train in the background (log in runs/main/train.log)
nohup caffeinate -i nice -n 10 .venv/bin/python -W ignore train.py --name main > runs/main/train.log 2>&1 &
.venv/bin/python -W ignore train.py --name main --resume     # continue after stopping
pkill -f "python -W ignore train.py"                         # stop (saves a checkpoint first)

# watch the newest checkpoint play live (q quit, n next level, space pause)
.venv/bin/python -W ignore watch.py --level 1-3 --speed 2

# videos
.venv/bin/python -W ignore replay.py records --level 1-1 --best  # current record (training does this itself)
.venv/bin/python -W ignore replay.py records --level 1-1         # every record, in order
.venv/bin/python -W ignore replay.py evals --level 1-1           # one run per checkpoint
.venv/bin/python -W ignore replay.py anomalies                   # glitch candidates
.venv/bin/python -W ignore ghosts.py --level 1-3 --at latest --count 300   # many attempts at once
.venv/bin/python -W ignore race_tas.py --attempts 50                       # fair race against the TAS
.venv/bin/python -W ignore replay.py race --level 1-1                      # ...and its video

# the website (docs/, served by GitHub Pages)
.venv/bin/python make_site.py

# graphs
.venv/bin/tensorboard --logdir runs/main/tb
```

Videos land in:

| folder | contents |
|---|---|
| `videos/records/` | the current record run per level, e.g. `1-3_16.66s.mp4`, replaced when beaten |
| `videos/ghosts/` | hundreds of attempts overlaid on the level |
| `videos/cool/` | secret pipes, glitch candidates, record progressions |
| `videos/other/` | everything else |
| `videos/races/` | fair races against a TAS (`replay.py race`) |
| `videos/best/` | the best of all of the above, in one place |

## Run outputs (`runs/main/`)

| file | contents |
|---|---|
| `checkpoints/` | model every 500k steps (thinned out over time) |
| `episodes/` | every training attempt's inputs + x/y, replayable frame-exactly |
| `frontier/` | restart points per level, plus `demo/` runs from `explore.py` |
| `records.jsonl` | each new best time per level, with its inputs |
| `anomalies.jsonl` | attempts with sudden position jumps |
| `evals.jsonl` | deterministic run on each level at every checkpoint |
| `explore.log` | what the wall explorer found |
| `status.json` | current stage, success rates, best times |

## Next

- Push the times down on every level.
- Check how it does on the held-out levels it has never seen.
- A "speed tech" phase: give it every-frame control, sub-pixel position and the
  game's frame-rule timing. Then tricks like the flagpole glitch or wall jumps become
  learnable, though nothing will tell it to use them.
