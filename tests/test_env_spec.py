import gymnasium as gym
import pytest

from env.spec import EnvSpec
from errors import UnsupportedEnvironmentError


def _spec(env_id: str) -> EnvSpec:
    env = gym.make(env_id)
    try:
        return EnvSpec.of(env, env_id)
    finally:
        env.close()


def test_reads_the_sizes_of_a_flat_discrete_environment():
    assert _spec("CartPole-v1") == EnvSpec("CartPole-v1", obs_size=4, action_count=2)


def test_rejects_continuous_actions():
    with pytest.raises(UnsupportedEnvironmentError, match="Pendulum-v1"):
        _spec("Pendulum-v1")
