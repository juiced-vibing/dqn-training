from torch import Tensor
from torch import nn


class QNetwork(nn.Module):
    def __init__(self, obs_size: int, action_count: int, hidden_size: int) -> None:
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_count),
        )

    def forward(self, obs: Tensor) -> Tensor:
        return self.layers(obs)
