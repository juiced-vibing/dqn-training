from torch import Tensor
from torch import nn


class QNetwork(nn.Module):
    """Map an observation to one Q-value per action."""

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
        """Return one Q-value per action for each observation."""
        return self.layers(obs)
