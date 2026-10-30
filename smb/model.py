"""Feature extractor for the block-grid observation.

Dense layers only: on this Mac's CPU (no MPS on macOS 13) matmuls run ~10x
faster than convolutions, and the grid is small enough to flatten.
"""
import gymnasium as gym
import torch
from torch import nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from smb.obs import N_OBJ

TILE_EMBED = 8


class GridExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict, features_dim=512):
        super().__init__(observation_space, features_dim)
        rows, cols = observation_space["tiles"].shape
        history = observation_space["objs"].shape[0]
        n_scalars = observation_space["scalars"].shape[0]
        self.tile_embed = nn.Embedding(256, TILE_EMBED)
        n_in = rows * cols * (TILE_EMBED + history * N_OBJ) + n_scalars
        self.net = nn.Sequential(
            nn.Linear(n_in, 512), nn.ReLU(),
            nn.Linear(512, features_dim), nn.ReLU(),
        )

    def forward(self, obs):
        tiles = self.tile_embed(obs["tiles"].long()).flatten(1)
        objs = nn.functional.one_hot(obs["objs"].long(), N_OBJ).float().flatten(1)
        return self.net(torch.cat([tiles, objs, obs["scalars"]], dim=1))
