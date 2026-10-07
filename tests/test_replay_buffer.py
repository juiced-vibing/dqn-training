import numpy as np

from dqn.agent.replay_buffer import ReplayBuffer


def test_overwrites_oldest_transitions_once_full():
    buffer = ReplayBuffer(capacity=5, obs_size=2, seed=0)
    for number in range(8):
        obs = np.full(2, number, dtype=np.float32)
        buffer.push(obs, action=0, reward=float(number), next_obs=obs, is_done=False)

    rewards = buffer.sample(500).rewards

    assert set(rewards.tolist()) == {3.0, 4.0, 5.0, 6.0, 7.0}
