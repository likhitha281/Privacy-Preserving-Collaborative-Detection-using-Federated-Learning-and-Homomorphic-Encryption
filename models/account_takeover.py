
import torch.nn as nn

from common.utils import get_weights, set_weights  # noqa: F401  (shared weight exchange)


class AccountTakeoverMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.network(x).squeeze(-1)


def build_model(input_dim):
    return AccountTakeoverMLP(input_dim)
