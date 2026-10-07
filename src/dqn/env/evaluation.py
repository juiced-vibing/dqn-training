from typing import TYPE_CHECKING
from typing import cast

import gymnasium as gym
from gymnasium.wrappers import RenderCollection
import numpy as np

from dqn.consts import EVALUATION_SEED_OFFSET
from dqn.utils.metrics import EvaluationMetrics


if TYPE_CHECKING:
    from dqn.agent.dqn_agent import DQNAgent


def _play_greedy_episode(agent: DQNAgent, env: gym.Env, seed: int) -> float:
    obs, _ = env.reset(seed=seed)
    ep_return, is_done = 0.0, False

    while not is_done:
        action = agent.select_action(obs, epsilon=0.0)
        obs, reward, terminated, truncated, _ = env.step(action)
        ep_return += float(reward)
        is_done = terminated or truncated

    return ep_return


def evaluate(agent: DQNAgent, env_id: str, episode_count: int, step: int) -> EvaluationMetrics:
    """Run the greedy policy on held-out seeds and summarize its returns."""
    env = gym.make(env_id)

    try:
        returns = [_play_greedy_episode(agent, env, seed=EVALUATION_SEED_OFFSET + episode) for episode in range(episode_count)]
    finally:
        env.close()

    return EvaluationMetrics(step=step, mean_return=float(np.mean(returns)), std_return=float(np.std(returns)))


def record_episode(agent: DQNAgent, env_id: str) -> list[np.ndarray]:
    """Play one greedy episode and return its rendered RGB frames."""
    env = RenderCollection(gym.make(env_id, render_mode="rgb_array"))

    try:
        _play_greedy_episode(agent, env, seed=EVALUATION_SEED_OFFSET)
        # RenderCollection always returns the collected frame list in rgb_array mode
        return cast("list[np.ndarray]", env.render())
    finally:
        env.close()
