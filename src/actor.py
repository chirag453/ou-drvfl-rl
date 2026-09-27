"""Lightweight gradient-trained actor for continuous control."""
import torch
import torch.nn as nn


class BoundedActor(nn.Module):
    def __init__(self, feat_dim, action_dim, action_max=1.0):
        super().__init__()
        self.action_max = action_max
        self.head = nn.Linear(feat_dim, action_dim)

    def forward(self, phi):
        return self.action_max * torch.tanh(self.head(phi))