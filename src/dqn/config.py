from pydantic import BaseModel
from pydantic import Field


class TrainingConfig(BaseModel, frozen=True):
    """Hyperparameters for one training run, defaults tuned for CartPole."""

    env_id: str = "CartPole-v1"
    seed: int = 1
    total_steps: int = Field(default=500_000, gt=0)
    learning_rate: float = Field(default=2.5e-4, gt=0)
    gamma: float = Field(default=0.99, ge=0, le=1)
    hidden_size: int = Field(default=128, gt=0)
    buf_cap: int = Field(default=10_000, gt=0)
    batch_size: int = Field(default=128, gt=0)
    learning_starts: int = Field(default=10_000, ge=0)
    train_frequency: int = Field(default=10, gt=0)
    target_update_interval: int = Field(default=500, gt=0)
    max_grad_norm: float = Field(default=10.0, gt=0)
    epsilon_start: float = Field(default=1.0, ge=0, le=1)
    epsilon_end: float = Field(default=0.05, ge=0, le=1)
    exploration_fraction: float = Field(default=0.5, gt=0, le=1)
    eval_interval: int = Field(default=10_000, gt=0)
    eval_episodes: int = Field(default=10, gt=0)
