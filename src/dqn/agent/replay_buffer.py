import msgspec
import numpy as np
import torch


class Batch(msgspec.Struct, frozen=True):
    """Sampled transitions as tensors, one row per transition."""

    obs: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    next_obs: torch.Tensor
    dones: torch.Tensor


class ReplayBuffer:
    """Fixed-size ring buffer of transitions, overwriting the oldest once full."""

    def __init__(self, capacity: int, obs_size: int, seed: int) -> None:
        self.__obs = np.zeros((capacity, obs_size), dtype=np.float32)
        self.__next_obs = np.zeros((capacity, obs_size), dtype=np.float32)

        self.__actions = np.zeros(capacity, dtype=np.int64)
        self.__rewards = np.zeros(capacity, dtype=np.float32)
        self.__dones = np.zeros(capacity, dtype=np.float32)

        self.__capacity = capacity
        self.__index = 0
        self.__size = 0

        self.__rng = np.random.default_rng(seed)

    def push(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        *,
        is_done: bool,
    ) -> None:
        """
        Store one transition.

        `is_done` must be the environment's `terminated` flag, not `truncated`,
        a time-limit cut still has future value to bootstrap from.
        """
        index = self.__index

        self.__obs[index] = obs
        self.__next_obs[index] = next_obs

        self.__actions[index] = action
        self.__rewards[index] = reward
        self.__dones[index] = is_done

        self.__index = (index + 1) % self.__capacity
        self.__size = min(self.__size + 1, self.__capacity)

    def sample(self, batch_size: int) -> Batch:
        """Draw `batch_size` transitions uniformly, with replacement."""
        indices = self.__rng.integers(0, self.__size, size=batch_size)
        return Batch(
            obs=torch.from_numpy(self.__obs[indices]),
            actions=torch.from_numpy(self.__actions[indices]),
            rewards=torch.from_numpy(self.__rewards[indices]),
            next_obs=torch.from_numpy(self.__next_obs[indices]),
            dones=torch.from_numpy(self.__dones[indices]),
        )

    def __len__(self) -> int:  # noqa: D105
        return self.__size
