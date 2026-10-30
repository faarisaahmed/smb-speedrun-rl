"""Gymnasium environment for SMB speed training.

One emulator per instance (stable-retro allows one per process, so use
SubprocVecEnv). Each episode starts at the beginning of a level drawn from
the current level set and ends on death, flagpole, or getting stuck.

Every episode's actions and Mario's x/y per step are recorded, so any run
can be replayed frame-exactly later (the emulator is deterministic).
"""
import base64
import gzip
import json
import os
import shutil
import time

import gymnasium as gym
import numpy as np
import stable_retro as retro

from smb.obs import COLS, N_OBJ, N_SCALARS, ROWS, observe
from smb.ram import RAM, x_pos, y_pos

STATES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "states")

# Button order in stable-retro's NES core.
BUTTONS = ["B", None, "SELECT", "START", "UP", "DOWN", "LEFT", "RIGHT", "A"]

ACTIONS = [
    [],
    ["RIGHT"],
    ["RIGHT", "A"],
    ["RIGHT", "B"],
    ["RIGHT", "A", "B"],
    ["A"],
    ["LEFT"],
    ["LEFT", "A"],
    ["LEFT", "B"],
    ["LEFT", "A", "B"],
    ["DOWN"],
    ["UP"],
]

# Object grid and scalars are given for the current and previous step so the
# agent can see which way enemies and platforms are moving.
HISTORY = 2

# Reward shaping. Progress is in pixels; a full level is ~3000-3500 px.
PROGRESS_SCALE = 1 / 20
TIME_PENALTY = 0.01      # per agent step, so faster finishes score higher
DEATH_PENALTY = 3.0
HURT_PENALTY = 1.0
FLAG_BONUS = 10.0
STUCK_STEPS = 150        # end the episode if max x hasn't improved in this many steps
MAX_STEPS = 3000

# A single 2-frame step normally moves Mario at most ~6 px. Bigger jumps
# usually mean a glitch (zip, clip, wrap) and get flagged for review.
ANOMALY_DX = 10

MIN_FREE_DISK = 10 * 1024 ** 3  # stop saving episodes below this much free space

# Frontier restarts (Go-Explore style): snapshot the emulator each time Mario
# reaches a new FRONTIER_BIN of distance, and start some episodes from those
# snapshots so hard sections get practiced far more often than once per
# full attempt. Snapshots are only taken on the ground and the last SAFE_STEPS
# before a death are thrown away, so restarts aren't already doomed (a longer
# margin would drop the take-off spot right before a hard jump). Restart points are picked by how
# often attempts from them fail to get PASS_BINS further, so practice goes
# where the agent struggles rather than to sections it has mastered.
FRONTIER_BIN = 64
SAFE_STEPS = 15
PASS_BINS = 4
HOPELESS_ATTEMPTS = 100   # a restart point with this many attempts and no passes gets replaced
SAVE_FRONTIER_EVERY = 100  # episodes between writing this env's restart points to disk
AREA_CHANGE_DX = 256   # pipes / bonus rooms: stop snapshotting after one
# Some jumps need inputs held exactly for many steps, which random exploration
# essentially never finds. Snapshots taken every few steps along one of the
# agent's own successful runs (frontier/<level>/demo/, see seed_frontier.py),
# mid-air included, are restart points too: from late in a jump only the last
# few inputs have to be right, and the failure weighting walks practice back
# toward the take-off as those get learned (the "backward" algorithm).
DEMO_DIR = "demo"


def load_state(name):
    with gzip.open(os.path.join(STATES_DIR, f"{name}.state"), "rb") as f:
        return f.read()


def button_array(buttons):
    a = np.zeros(len(BUTTONS), dtype=np.int8)
    for b in buttons:
        a[BUTTONS.index(b)] = 1
    return a


class EpisodeRecorder:
    """Buffers finished episodes and flushes them to compressed .npz chunks."""

    def __init__(self, out_dir, rank, flush_every=200):
        self.out_dir = out_dir
        self.rank = rank
        self.flush_every = flush_every
        self.episodes = []
        os.makedirs(out_dir, exist_ok=True)
        # Continue numbering after a resume instead of overwriting old chunks.
        prefix = f"env{rank:02d}_"
        done = [int(f[len(prefix):-len(".npz")]) for f in os.listdir(out_dir)
                if f.startswith(prefix) and f.endswith(".npz")]
        self.chunk = max(done, default=-1) + 1

    def add(self, actions, xs, ys, meta):
        self.episodes.append((np.array(actions, np.uint8), np.array(xs, np.uint16),
                              np.array(ys, np.int16), meta))
        if len(self.episodes) >= self.flush_every:
            self.flush()

    def flush(self):
        if not self.episodes:
            return
        if shutil.disk_usage(self.out_dir).free < MIN_FREE_DISK:
            self.episodes = []
            return
        actions, xs, ys, metas = zip(*self.episodes)
        lengths = np.array([len(a) for a in actions], np.int32)
        path = os.path.join(self.out_dir, f"env{self.rank:02d}_{self.chunk:05d}.npz")
        np.savez_compressed(
            path,
            lengths=lengths,
            actions=np.concatenate(actions),
            x=np.concatenate(xs),
            y=np.concatenate(ys),
            meta=np.array(json.dumps(metas)),
        )
        self.chunk += 1
        self.episodes = []


class SMBEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, levels, frame_skip=2, rank=0, record_dir=None, seed=0,
                 mid_start_prob=0.0, frontier_dir=None):
        super().__init__()
        self.env = retro.make("SuperMarioBros-Nes-v0", state=retro.State.NONE,
                              use_restricted_actions=retro.Actions.ALL, render_mode=None)
        self.emu = self.env.unwrapped
        self.frame_skip = frame_skip
        self.rank = rank
        self.rng = np.random.default_rng(seed + rank)
        self.states = {}
        self.mid_start_prob = mid_start_prob
        self.frontier_dir = frontier_dir
        self.frontier = {}   # level -> {bin: emulator state}
        self.demo = {}       # level -> {("demo", step): emulator state}, never replaced
        self.bin_stats = {}  # level -> {bin or demo key: [attempts, passes]} for restarts from it
        self.set_levels(levels)
        self.button_arrays = [button_array(b) for b in ACTIONS]
        self.action_space = gym.spaces.Discrete(len(ACTIONS))
        self.observation_space = gym.spaces.Dict({
            "tiles": gym.spaces.Box(0, 255, (ROWS, COLS), np.float32),
            "objs": gym.spaces.Box(0, N_OBJ - 1, (HISTORY, ROWS, COLS), np.float32),
            "scalars": gym.spaces.Box(-np.inf, np.inf, (HISTORY * N_SCALARS,), np.float32),
        })
        self.recorder = EpisodeRecorder(record_dir, rank) if record_dir else None
        self.total_steps = 0
        self.episode_id = 0
        self.screen = None

    def set_levels(self, levels, weights=None):
        self.levels = list(levels)
        w = np.ones(len(self.levels)) if weights is None else np.asarray(weights, float)
        self.weights = w / w.sum()
        for name in self.levels:
            if name not in self.states:
                self.states[name] = load_state(name)

    def load_frontier(self, level):
        self.frontier[level] = {}
        d = os.path.join(self.frontier_dir or "", level)
        if self.frontier_dir and os.path.isdir(d):
            for f in os.listdir(d):
                if f.startswith(".") or not f.endswith(".state"):
                    continue
                with gzip.open(os.path.join(d, f), "rb") as fh:
                    self.frontier[level][int(f.split(".")[0])] = fh.read()
        self.load_demo(level)

    def load_demo(self, level):
        """(Re)load demo restart points; explore.py adds them while training runs."""
        self.demo[level] = {}
        d = os.path.join(self.frontier_dir or "", level, DEMO_DIR)
        if self.frontier_dir and os.path.isdir(d):
            for f in os.listdir(d):
                if f.endswith(".state"):
                    with gzip.open(os.path.join(d, f), "rb") as fh:
                        self.demo[level][(DEMO_DIR, int(f.split(".")[0]))] = fh.read()

    def pick_frontier(self, level):
        if level not in self.frontier:
            self.load_frontier(level)
        points = {**self.frontier[level], **self.demo[level]}
        if not points or self.rng.random() >= self.mid_start_prob:
            return None
        keys = list(points)
        stats = self.bin_stats.setdefault(level, {})
        fail_rate = np.array([1 - (stats.get(k, [0, 0])[1] + 1) / (stats.get(k, [0, 0])[0] + 2) for k in keys])
        hopeless = np.array([self.is_hopeless(level, k) for k in keys])
        w = 0.05 + fail_rate * np.where(hopeless, 0.2, 1.0)
        k = keys[self.rng.choice(len(keys), p=w / w.sum())]
        self.start_key = k
        return points[k]

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        level = options.get("level") or self.levels[self.rng.choice(len(self.levels), p=self.weights)]
        self.level = level
        if level not in self.states:
            self.states[level] = load_state(level)
        start = options.get("state")
        self.start_key = None
        if start is None and "level" not in options:
            start = self.pick_frontier(level)
        self.start_state = start
        self.emu.initial_state = start or self.states[level]
        obs, _ = self.env.reset()
        self.screen = obs
        ram = self.emu.get_ram()
        self.start_x = self.prev_x = self.max_x = x_pos(ram)
        self.prev_powerup = int(ram[RAM.POWERUP])
        self.steps = 0
        self.frames = 0
        self.since_progress = 0
        self.actions, self.xs, self.ys = [], [], []
        self.anomalies = []
        self.pending = []   # (step, bin, state) snapshots awaiting commit
        self.next_bin = self.max_x // FRONTIER_BIN + 1
        self.area_changed = False
        tiles, objs, scalars = observe(ram)
        self.obj_hist = [objs] * HISTORY
        self.scalar_hist = [scalars] * HISTORY
        return self.make_obs(tiles), {"level": level}

    def make_obs(self, tiles):
        return {"tiles": tiles, "objs": np.stack(self.obj_hist),
                "scalars": np.concatenate(self.scalar_hist)}

    def step(self, action):
        buttons = self.button_arrays[action]
        for _ in range(self.frame_skip):
            obs, _, _, _, _ = self.env.step(buttons)
            self.frames += 1
            ram = self.emu.get_ram()
            # Stop early on the frame the outcome is decided, so recorded
            # frame counts are exact.
            if ram[RAM.PLAYER_STATE] in (RAM.DYING, RAM.DEAD) or ram[RAM.Y_PAGE] > 1 \
                    or ram[RAM.FLOAT_STATE] == RAM.FLAGPOLE:
                break
        self.screen = obs
        self.steps += 1
        self.total_steps += 1

        x, y = x_pos(ram), y_pos(ram)
        dx = x - self.prev_x
        if abs(dx) > ANOMALY_DX:
            self.anomalies.append([self.steps, int(dx)])
        if abs(dx) >= AREA_CHANGE_DX:
            self.area_changed = True
        self.prev_x = x
        # Only snapshot with Mario standing on something: a mid-air snapshot can
        # be a fall that's already doomed but takes longer than SAFE_STEPS.
        on_ground = ram[RAM.FLOAT_STATE] == 0 and ram[RAM.Y_PAGE] == 1
        if self.mid_start_prob and not self.area_changed and on_ground and x // FRONTIER_BIN >= self.next_bin:
            self.next_bin = x // FRONTIER_BIN + 1
            self.pending.append((self.steps, x // FRONTIER_BIN, self.emu.em.get_state()))

        reward = float(np.clip(dx, -15, 15)) * PROGRESS_SCALE - TIME_PENALTY
        if x > self.max_x:
            self.max_x = x
            self.since_progress = 0
        else:
            self.since_progress += 1

        powerup = int(ram[RAM.POWERUP])
        if powerup < self.prev_powerup:
            reward -= HURT_PENALTY
        self.prev_powerup = powerup

        died = ram[RAM.PLAYER_STATE] in (RAM.DYING, RAM.DEAD) or ram[RAM.Y_PAGE] > 1
        flag = ram[RAM.FLOAT_STATE] == RAM.FLAGPOLE
        terminated = died or flag
        truncated = not terminated and (self.since_progress >= STUCK_STEPS or self.steps >= MAX_STEPS)
        if died:
            reward -= DEATH_PENALTY
        if flag:
            reward += FLAG_BONUS

        self.actions.append(int(action))
        self.xs.append(x)
        self.ys.append(y)

        info = {"level": self.level, "x": x}
        if terminated or truncated:
            outcome = "flag" if flag else "death" if died else "stuck"
            safe_bin = self.commit_frontier(self.steps - SAFE_STEPS if died else self.steps)
            if self.start_key is not None:
                # A pass means standing safely PASS_BINS further on; just getting
                # that far (e.g. falling past a ledge into a pit) doesn't count.
                stat = self.bin_stats[self.level].setdefault(self.start_key, [0, 0])
                stat[0] += 1
                stat[1] += int(flag or safe_bin >= self.start_x // FRONTIER_BIN + PASS_BINS)
            meta = {
                "id": self.episode_id, "rank": self.rank, "level": self.level,
                "outcome": outcome, "frames": self.frames, "steps": self.steps,
                "max_x": self.max_x, "start_x": self.start_x,
                "env_steps": self.total_steps, "time": time.time(),
                "frame_skip": self.frame_skip, "anomalies": self.anomalies,
                "from_start": self.start_state is None,
                "demo_start": isinstance(self.start_key, tuple),
            }
            if self.recorder:
                self.recorder.add(self.actions, self.xs, self.ys, meta)
            self.episode_id += 1
            info["episode_meta"] = meta
            info["episode_actions"] = list(self.actions)
            if self.start_state is not None and self.anomalies:
                # Mid-level runs can only be replayed from their start snapshot.
                info["start_state"] = base64.b64encode(self.start_state).decode()
        tiles, objs, scalars = observe(ram)
        self.obj_hist = self.obj_hist[1:] + [objs]
        self.scalar_hist = self.scalar_hist[1:] + [scalars]
        return self.make_obs(tiles), reward, terminated, truncated, info

    def is_hopeless(self, level, b):
        attempts, passes = self.bin_stats.get(level, {}).get(b, [0, 0])
        return attempts >= HOPELESS_ATTEMPTS and passes == 0

    def commit_frontier(self, last_safe_step):
        """Keep this episode's safe snapshots; returns the furthest safe bin (-1 if none)."""
        if not self.pending:
            return -1
        if self.level not in self.frontier:
            self.load_frontier(self.level)
        frontier = self.frontier[self.level]
        stats = self.bin_stats.setdefault(self.level, {})
        safe_bin = -1
        for step, b, state in self.pending:
            if step > last_safe_step:
                continue
            safe_bin = max(safe_bin, b)
            if b not in frontier or self.is_hopeless(self.level, b) or self.rng.random() < 0.5:
                frontier[b] = state
                stats.pop(b, None)   # new snapshot, fresh stats
        self.pending = []
        if self.frontier_dir and self.episode_id % SAVE_FRONTIER_EVERY == 0:
            self.save_frontier()
        return safe_bin

    def save_frontier(self):
        # Shared across envs (last writer wins); seeds restarts after a resume.
        for level, points in self.frontier.items():
            d = os.path.join(self.frontier_dir, level)
            os.makedirs(d, exist_ok=True)
            for b, state in points.items():
                tmp = os.path.join(d, f".{b}.{self.rank}.tmp")
                with gzip.open(tmp, "wb") as f:
                    f.write(state)
                os.replace(tmp, os.path.join(d, f"{b}.state"))

    def render(self):
        return self.screen

    def close(self):
        if self.recorder:
            self.recorder.flush()
        self.env.close()
