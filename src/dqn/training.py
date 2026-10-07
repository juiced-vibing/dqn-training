from typing import TYPE_CHECKING

import gymnasium as gym
from gymnasium.spaces import Box
from gymnasium.spaces import Discrete
from loguru import logger
import torch

from dqn.agent.dqn_agent import DQNAgent
from dqn.agent.replay_buffer import ReplayBuffer
from dqn.env.evaluation import evaluate
from dqn.errors import UnsupportedEnvironmentError
from dqn.utils.metrics import EpisodeMetrics
from dqn.utils.metrics import Metrics
from dqn.utils.schedules import linear_schedule


if TYPE_CHECKING:
    from collections.abc import Callable

    from dqn.config import TrainingConfig


def run_training(config: TrainingConfig, on_metrics: Callable[[Metrics], None]) -> DQNAgent:
    """Train an agent and return it holding the weights of its best evaluation."""
    torch.manual_seed(config.seed)

    env = gym.make(config.env_id)
    obs_space, action_space = env.observation_space, env.action_space
    if not isinstance(obs_space, Box) or len(obs_space.shape) != 1 or not isinstance(action_space, Discrete):
        env.close()
        raise UnsupportedEnvironmentError(config.env_id)

    obs_size = obs_space.shape[0]
    agent = DQNAgent(obs_size, int(action_space.n), config)
    buf = ReplayBuffer(config.buf_cap, obs_size, config.seed)

    exploration_steps = int(config.exploration_fraction * config.total_steps)
    best_return, best_weights = float("-inf"), agent.copy_weights()

    try:
        obs, _ = env.reset(seed=config.seed)
        ep_return, ep_len = 0.0, 0

        for step in range(1, config.total_steps + 1):
            epsilon = linear_schedule(config.epsilon_start, config.epsilon_end, exploration_steps, step)

            action = agent.select_action(obs, epsilon)
            next_obs, reward, terminated, truncated, _ = env.step(action)

            buf.push(obs, action, float(reward), next_obs, is_done=terminated)

            obs = next_obs
            ep_return += float(reward)
            ep_len += 1

            if terminated or truncated:
                on_metrics(EpisodeMetrics(step, ep_return, ep_len, epsilon))
                # unseeded reset continues the environment's own rng from the first seed
                obs, _ = env.reset()
                ep_return, ep_len = 0.0, 0

            if step >= config.learning_starts and step % config.train_frequency == 0:
                on_metrics(agent.learn(buf.sample(config.batch_size), step))
            if step % config.target_update_interval == 0:
                agent.sync_target()
            if step % config.eval_interval == 0 or step == config.total_steps:
                evaluation = evaluate(agent, config.env_id, config.eval_episodes, step)
                on_metrics(evaluation)

                if evaluation.mean_return > best_return:
                    best_return, best_weights = evaluation.mean_return, agent.copy_weights()
                    logger.debug("new best evaluation {} at step {}", best_return, step)
    finally:
        env.close()

    agent.load_weights(best_weights)
    return agent
