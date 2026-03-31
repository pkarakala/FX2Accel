from __future__ import annotations

import torch
import torch.nn as nn


class SimpleMLP(nn.Module):
    def __init__(self, in_features: int = 4, hidden: int = 4, out_features: int = 4) -> None:
        super().__init__()
        self.lin = nn.Linear(in_features, hidden)
        self.relu = nn.ReLU()
        self.out = nn.Linear(hidden, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.lin(x)
        x = self.relu(x)
        x = self.out(x)
        return x
