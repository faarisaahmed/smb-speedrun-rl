"""PPO plus imitation of the agent's own explorer-found demos.

Some moves (e.g. a full-height jump from a 3-step take-off window) are almost
never sampled by the policy, so PPO never gets a success to learn from. After
each PPO update, a few extra gradient steps raise the log-probability of the
inputs explore.py used on its successful runs (frontier/<level>/demo/*.npz).
"""
import glob
import os

import numpy as np
import torch
from stable_baselines3 import PPO

SIL_COEF = 0.1
SIL_STEPS = 4       # gradient steps per PPO update (more hurt 1-1: 0.99 -> 0.75 clears)
SIL_BATCH = 256


def load_demos(frontier_dir, levels):
    """All demo (observation, action) pairs for these levels, or None."""
    parts = []
    for level in levels:
        for path in glob.glob(os.path.join(frontier_dir, level, "demo", "*.npz")):
            with np.load(path) as d:
                parts.append({k: d[k] for k in d.files})
    if not parts:
        return None
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


class SILPPO(PPO):
    demos = None   # set by the training callback; not saved with the model

    def train(self):
        super().train()
        if self.demos is None:
            return
        self.policy.set_training_mode(True)
        n = len(self.demos["actions"])
        losses = []
        for _ in range(SIL_STEPS):
            idx = np.random.randint(n, size=min(SIL_BATCH, n))
            obs = {k: torch.as_tensor(v[idx], device=self.device) for k, v in self.demos.items() if k != "actions"}
            actions = torch.as_tensor(self.demos["actions"][idx], device=self.device).long()
            _, log_prob, _ = self.policy.evaluate_actions(obs, actions)
            loss = -SIL_COEF * log_prob.mean()
            self.policy.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()
            losses.append(-log_prob.mean().item())
        self.logger.record("train/sil_nll", float(np.mean(losses)))
