"""Train a PPO agent on SMB with an automatic level curriculum.

Stage 0: 1-1 only. Stage 1: 1-1 + 1-3. Stage 2: every outdoor training level,
sampled more often where the agent is weaker. A stage advances once every
level in it is cleared in at least ADVANCE_RATE of its recent episodes.
HOLDOUT levels are never trained on; they're only evaluated, to measure
whether the agent has learned to play levels it has never seen.

Outputs under runs/<name>/:
  checkpoints/     model every --ckpt-every steps
  episodes/        every training episode (actions + x/y), replayable
  records.jsonl    each new fastest flag time per level, with actions
  anomalies.jsonl  episodes with suspicious position jumps (glitch candidates)
  evals.jsonl      deterministic run on each level at every checkpoint
  status.json      current stage/levels/stats, read by watch.py
  tb/              TensorBoard logs
"""
import argparse
import json
import os
import signal
import subprocess
import sys
import time
from collections import defaultdict, deque

import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor

from smb.env import SMBEnv
from smb.model import GridExtractor
from smb.sil import SILPPO, load_demos

STAGES = [
    ["1-1"],
    ["1-1", "1-3"],
    ["1-1", "1-3", "2-1", "2-3", "3-1", "3-2", "3-3", "4-1", "4-3",
     "5-1", "5-2", "6-2", "6-3", "7-1", "7-3", "8-1", "8-3"],
]
HOLDOUT = ["5-3", "6-1", "8-2"]
# Keep every checkpoint early (the interesting learning), then thin out.
KEEP_ALL_UNTIL = 10_000_000
KEEP_EVERY = 5_000_000
KEEP_RECENT = 3
EXPLORE_EVERY = 4   # checkpoints between explore.py runs (walls the policy keeps dying at)
ADVANCE_RATE = 0.6
WINDOW = 200


def make_env(levels, rank, record_dir, frame_skip, mid_start_prob=0.0, frontier_dir=None):
    def thunk():
        return SMBEnv(levels, frame_skip=frame_skip, rank=rank, record_dir=record_dir,
                      mid_start_prob=mid_start_prob, frontier_dir=frontier_dir)
    return thunk


def append_jsonl(path, obj):
    with open(path, "a") as f:
        f.write(json.dumps(obj) + "\n")


def evaluate(model, level, frame_skip):
    env = DummyVecEnv([lambda: SMBEnv([level], frame_skip=frame_skip, rank=99)])
    obs = env.reset()
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, infos = env.step(action)
        if done[0]:
            env.close()
            meta = infos[0]["episode_meta"]
            return meta, infos[0].get("episode_actions")


class CurriculumCallback(BaseCallback):
    def __init__(self, run_dir, stage, ckpt_every, frame_skip):
        super().__init__()
        self.run_dir = run_dir
        self.stage = stage
        self.ckpt_every = ckpt_every
        self.frame_skip = frame_skip
        self.recent = defaultdict(lambda: deque(maxlen=WINDOW))  # level -> [(cleared, max_x, frames)]
        self.best = {}
        self.next_ckpt = 0
        self.next_log = 0
        self.started = time.time()
        self.explorer = None
        self.ckpts = 0
        self.renders = {}   # level -> (render process, record frames it's rendering)
        os.makedirs(os.path.join(run_dir, "checkpoints"), exist_ok=True)
        best_path = os.path.join(run_dir, "records.jsonl")
        if os.path.exists(best_path):
            for line in open(best_path):
                r = json.loads(line)
                self.best[r["level"]] = min(self.best.get(r["level"], 1e9), r["frames"])

    def _on_training_start(self):
        self.next_ckpt = (self.num_timesteps // self.ckpt_every + 1) * self.ckpt_every
        self.apply_levels()
        self.load_demos()

    def load_demos(self):
        self.model.demos = load_demos(os.path.join(self.run_dir, "frontier"), self.levels())
        n = 0 if self.model.demos is None else len(self.model.demos["actions"])
        print(f"[demos] {n} demo steps for imitation", flush=True)

    def levels(self):
        return STAGES[self.stage]

    def success(self, level):
        r = self.recent[level]
        return sum(c for c, _, _ in r) / len(r) if r else 0.0

    def apply_levels(self):
        levels = self.levels()
        # Spend more time on levels the agent is bad at.
        weights = [1.0 - 0.8 * self.success(l) for l in levels]
        self.training_env.env_method("set_levels", levels, weights)

    def _on_step(self):
        for info in self.locals["infos"]:
            meta = info.get("episode_meta")
            if meta is None:
                continue
            level = meta["level"]
            if meta["anomalies"]:
                append_jsonl(os.path.join(self.run_dir, "anomalies.jsonl"), {
                    **meta, "timesteps": self.num_timesteps, "actions": info.get("episode_actions"),
                    "start_state": info.get("start_state")})
            if not meta["from_start"]:
                continue   # frontier restarts don't count toward records or the curriculum
            cleared = meta["outcome"] == "flag"
            self.recent[level].append((cleared, meta["max_x"], meta["frames"]))
            if cleared and meta["frames"] < self.best.get(level, 1e9):
                self.best[level] = meta["frames"]
                append_jsonl(os.path.join(self.run_dir, "records.jsonl"), {
                    **meta, "timesteps": self.num_timesteps, "actions": info["episode_actions"]})
                print(f"[record] {level}: {meta['frames']} frames "
                      f"({meta['frames'] / 60.0988:.2f}s) at {self.num_timesteps:,} steps", flush=True)

        if self.num_timesteps >= self.next_log:
            self.next_log = self.num_timesteps + 20_000
            self.log_stats()
            self.maybe_advance()

        if self.num_timesteps >= self.next_ckpt:
            self.next_ckpt += self.ckpt_every
            self.checkpoint()
        return True

    def maybe_advance(self):
        if self.stage + 1 >= len(STAGES):
            self.apply_levels()
            return
        ready = all(len(self.recent[l]) >= WINDOW // 2 and self.success(l) >= ADVANCE_RATE
                    for l in self.levels())
        if ready:
            self.stage += 1
            print(f"[curriculum] advancing to stage {self.stage}: {self.levels()}", flush=True)
            self.load_demos()
            append_jsonl(os.path.join(self.run_dir, "curriculum.jsonl"),
                         {"stage": self.stage, "timesteps": self.num_timesteps, "time": time.time()})
            self.checkpoint()
        self.apply_levels()

    def render_records(self):
        """Keep videos/records/<level>_<time>s.mp4 at the current record (replay.py --best)."""
        for level, frames in self.best.items():
            proc, rendered = self.renders.get(level, (None, None))
            if rendered == frames or (proc is not None and proc.poll() is None):
                continue
            proc = subprocess.Popen(
                ["nice", "-n", "15", sys.executable, "-W", "ignore", "replay.py", "records",
                 "--run", os.path.basename(self.run_dir), "--level", level, "--best"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            self.renders[level] = (proc, frames)

    def log_stats(self):
        self.render_records()
        for level, r in self.recent.items():
            if not r:
                continue
            self.logger.record(f"success/{level}", self.success(level))
            self.logger.record(f"max_x/{level}", float(np.mean([m for _, m, _ in r])))
            if level in self.best:
                self.logger.record(f"best_frames/{level}", self.best[level])
        self.logger.record("curriculum/stage", self.stage)
        status = {
            "stage": self.stage, "levels": self.levels(), "holdout": HOLDOUT,
            "timesteps": self.num_timesteps, "hours": (time.time() - self.started) / 3600,
            "success": {l: round(self.success(l), 3) for l in self.recent},
            "best_frames": self.best,
        }
        with open(os.path.join(self.run_dir, "status.json"), "w") as f:
            json.dump(status, f, indent=1)

    def checkpoint(self):
        path = os.path.join(self.run_dir, "checkpoints", f"step_{self.num_timesteps:010d}")
        self.model.save(path)
        self.prune_checkpoints()
        eval_levels = self.levels() + (HOLDOUT if self.stage == len(STAGES) - 1 else [])
        for level in eval_levels:
            meta, actions = evaluate(self.model, level, self.frame_skip)
            append_jsonl(os.path.join(self.run_dir, "evals.jsonl"), {
                "checkpoint": os.path.basename(path), "timesteps": self.num_timesteps,
                "holdout": level in HOLDOUT, **meta, "actions": actions})
            self.logger.record(f"eval_max_x/{level}", meta["max_x"])
        print(f"[ckpt] {self.num_timesteps:,} steps, stage {self.stage}, "
              f"success {{{', '.join(f'{l}: {self.success(l):.2f}' for l in self.levels())}}}", flush=True)
        self.run_explorer()

    def run_explorer(self):
        """Every EXPLORE_EVERY checkpoints, look for walls in the background (explore.py)."""
        self.ckpts += 1
        if self.explorer is not None:
            if self.explorer.poll() is None:
                return
            for level in self.levels():
                self.training_env.env_method("load_demo", level)
            self.load_demos()
            self.explorer = None
        if self.ckpts % EXPLORE_EVERY:
            return
        log = open(os.path.join(self.run_dir, "explore.log"), "a")
        self.explorer = subprocess.Popen(
            ["nice", "-n", "15", sys.executable, "-W", "ignore", "explore.py",
             "--run", os.path.basename(self.run_dir), "--hours", "3", "--levels", *self.levels()],
            stdout=log, stderr=subprocess.STDOUT)


    def prune_checkpoints(self):
        d = os.path.join(self.run_dir, "checkpoints")
        ckpts = sorted(f for f in os.listdir(d) if f.endswith(".zip"))
        for f in ckpts[:-KEEP_RECENT]:
            step = int(f[len("step_"):-len(".zip")])
            # Keep the checkpoint closest after each KEEP_EVERY boundary.
            boundary = step // KEEP_EVERY * KEEP_EVERY
            first_after = step - boundary < self.ckpt_every
            if step > KEEP_ALL_UNTIL and not first_after:
                os.remove(os.path.join(d, f))


def stop_gracefully(signum, frame):
    # Background jobs ignore SIGINT, so make `pkill` (SIGTERM) save a checkpoint on the way out.
    raise KeyboardInterrupt


def main():
    signal.signal(signal.SIGTERM, stop_gracefully)
    p = argparse.ArgumentParser()
    p.add_argument("--name", default="main")
    p.add_argument("--envs", type=int, default=16)
    p.add_argument("--steps", type=int, default=2_000_000_000)
    p.add_argument("--frame-skip", type=int, default=2)
    p.add_argument("--ckpt-every", type=int, default=500_000)
    p.add_argument("--device", default="auto")
    p.add_argument("--mid-start", type=float, default=0.5,
                   help="fraction of episodes started from a saved frontier snapshot")
    p.add_argument("--resume", action="store_true", help="continue from the latest checkpoint in the run")
    args = p.parse_args()

    run_dir = os.path.join("runs", args.name)
    os.makedirs(run_dir, exist_ok=True)
    ckpt_dir = os.path.join(run_dir, "checkpoints")

    stage = 0
    status_path = os.path.join(run_dir, "status.json")
    if args.resume and os.path.exists(status_path):
        stage = json.load(open(status_path))["stage"]

    venv = SubprocVecEnv([make_env(STAGES[stage], i, os.path.join(run_dir, "episodes"), args.frame_skip,
                                   args.mid_start, os.path.join(run_dir, "frontier"))
                          for i in range(args.envs)], start_method="spawn")
    venv = VecMonitor(venv)

    device = args.device
    if device == "auto":
        device = "cpu"  # the grid MLP is small; CPU beats MPS for it anyway
    torch.set_num_threads(2)  # leave cores for the emulator processes

    ckpts = sorted(os.listdir(ckpt_dir)) if args.resume and os.path.isdir(ckpt_dir) else []
    if ckpts:
        latest = os.path.join(ckpt_dir, ckpts[-1])
        print("resuming from", latest)
        model = SILPPO.load(latest, env=venv, device=device, tensorboard_log=os.path.join(run_dir, "tb"))
    else:
        model = SILPPO(
            "MultiInputPolicy", venv, device=device,
            policy_kwargs=dict(features_extractor_class=GridExtractor,
                               net_arch=dict(pi=[256], vf=[256])),
            n_steps=512, batch_size=1024, n_epochs=4,
            learning_rate=2.5e-4, gamma=0.99, gae_lambda=0.95,
            clip_range=0.2, ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5,
            tensorboard_log=os.path.join(run_dir, "tb"), verbose=0,
        )

    callback = CurriculumCallback(run_dir, stage, args.ckpt_every, args.frame_skip)
    try:
        model.learn(args.steps, callback=callback, reset_num_timesteps=not ckpts, tb_log_name="ppo")
    finally:
        if callback.model is not None:
            callback.checkpoint()
        venv.close()


if __name__ == "__main__":
    main()
