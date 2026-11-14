from copy import deepcopy
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch.nn.functional import mse_loss
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam

from agent.network import QNetwork
from utils.metrics import LearnMetrics


if TYPE_CHECKING:
    from agent.replay_buffer import Batch
    from config import TrainingConfig
    from env.spec import EnvSpec


class DQNAgent:
    def __init__(self, spec: EnvSpec, config: TrainingConfig) -> None:
        self.__online = QNetwork(spec.obs_size, spec.action_count, config.hidden_size)
        self.__target = deepcopy(self.__online)

        self.__optimizer = Adam(self.__online.parameters(), lr=config.learning_rate)

        self.__gamma = config.gamma
        self.__max_grad_norm = config.max_grad_norm

        self.__action_count = spec.action_count

        self.__rng = np.random.default_rng(config.seed)

    def sync_target(self) -> None:
        self.__target.load_state_dict(self.__online.state_dict())

    def copy_weights(self) -> dict[str, torch.Tensor]:
        # cloned, so the copy stays as it is while training continues
        return {name: tensor.clone() for name, tensor in self.__online.state_dict().items()}

    def load_weights(self, weights: dict[str, torch.Tensor]) -> None:
        self.__online.load_state_dict(weights)
        self.__target.load_state_dict(weights)

    def q_values(self, observation: np.ndarray) -> np.ndarray:
        with torch.inference_mode():
            return self.__online(torch.as_tensor(observation)).numpy()

    def select_action(self, observation: np.ndarray, epsilon: float) -> int:
        if self.__rng.random() < epsilon:
            return int(self.__rng.integers(self.__action_count))

        return int(self.q_values(observation).argmax())

    def learn(self, batch: Batch, step: int) -> LearnMetrics:
        with torch.no_grad():
            # online net picks the next action, target net scores it, curbing overestimation
            next_actions = self.__online(batch.next_obs).argmax(dim=1, keepdim=True)
            next_q = self.__target(batch.next_obs).gather(1, next_actions).squeeze(1)
            target_q = batch.rewards + self.__gamma * (1.0 - batch.dones) * next_q

        predicted_q = self.__online(batch.obs).gather(1, batch.actions.unsqueeze(1)).squeeze(1)
        loss = mse_loss(predicted_q, target_q)

        self.__optimizer.zero_grad(set_to_none=True)
        loss.backward()
        clip_grad_norm_(self.__online.parameters(), self.__max_grad_norm)
        self.__optimizer.step()

        return LearnMetrics(step=step, loss=loss.item(), mean_q=predicted_q.mean().item())
