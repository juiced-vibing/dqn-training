"""Probe tests: tiny problems with a known optimal Q, each isolating one part of DQN."""

import numpy as np
import pytest

from dqn.agent.dqn_agent import DQNAgent
from dqn.agent.replay_buffer import ReplayBuffer
from dqn.config import TrainingConfig


_STATE_A = np.array([0.0], dtype=np.float32)
_STATE_B = np.array([1.0], dtype=np.float32)


def _make_agent(action_count: int = 1, gamma: float = 0.99) -> DQNAgent:
    config = TrainingConfig(learning_rate=1e-2, hidden_size=32, gamma=gamma, seed=0)
    return DQNAgent(obs_size=1, action_count=action_count, config=config)


def _train(agent: DQNAgent, buffer: ReplayBuffer, updates: int) -> None:
    for step in range(updates):
        agent.learn(buffer.sample(32), step)
        if step % 50 == 0:
            agent.sync_target()


def test_bootstraps_discounted_value_through_non_terminal_step():
    # a -> b with reward 0, b -> end with reward 1, so q(b) = 1 and q(a) = gamma
    agent = _make_agent(gamma=0.5)
    buffer = ReplayBuffer(capacity=64, obs_size=1, seed=0)
    buffer.push(_STATE_A, 0, 0.0, _STATE_B, is_done=False)
    buffer.push(_STATE_B, 0, 1.0, _STATE_B, is_done=True)

    _train(agent, buffer, updates=1_000)

    assert agent.q_values(_STATE_B)[0] == pytest.approx(1.0, abs=0.05)
    assert agent.q_values(_STATE_A)[0] == pytest.approx(0.5, abs=0.05)


def test_learns_mean_not_median_of_stochastic_reward():
    # rare large outcomes must move q, like rare pole-falls in cartpole
    # huber loss lands near the median (0.25 here) instead of the mean (2.0)
    agent = _make_agent()
    buffer = ReplayBuffer(capacity=10, obs_size=1, seed=0)
    for reward in [0.0] * 8 + [10.0] * 2:
        buffer.push(_STATE_A, 0, reward, _STATE_A, is_done=True)

    _train(agent, buffer, updates=1_000)

    assert agent.q_values(_STATE_A)[0] == pytest.approx(2.0, abs=0.3)


def test_greedy_action_picks_higher_reward_action():
    agent = _make_agent(action_count=2)
    buffer = ReplayBuffer(capacity=64, obs_size=1, seed=0)
    buffer.push(_STATE_A, 0, -1.0, _STATE_A, is_done=True)
    buffer.push(_STATE_A, 1, 1.0, _STATE_A, is_done=True)

    _train(agent, buffer, updates=500)

    assert agent.select_action(_STATE_A, epsilon=0.0) == 1
